// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package gcpcache

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/filterapi"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
)

// -----------------------------------------------------------------------
// Fake GCPAuthHandler
// -----------------------------------------------------------------------

type fakeGCPAuth struct {
	token   string
	region  string
	project string
}

func (f *fakeGCPAuth) Do(_ context.Context, _ map[string]string, _ []byte) ([]internalapi.Header, error) {
	return nil, nil
}

func (f *fakeGCPAuth) GCPTokenSource() oauth2.TokenSource {
	return oauth2.StaticTokenSource(&oauth2.Token{AccessToken: f.token})
}
func (f *fakeGCPAuth) GCPRegion() string  { return f.region }
func (f *fakeGCPAuth) GCPProject() string { return f.project }

var _ filterapi.GCPAuthHandler = (*fakeGCPAuth)(nil)

// -----------------------------------------------------------------------
// Message builders
// -----------------------------------------------------------------------

func ephemeralFields() *openai.AnthropicContentFields {
	return &openai.AnthropicContentFields{
		CacheControl: anthropic.CacheControlEphemeralParam{
			Type: constant.ValueOf[constant.Ephemeral](),
		},
	}
}

func ephemeralFieldsWithTTL(ttl anthropic.CacheControlEphemeralTTL) *openai.AnthropicContentFields {
	return &openai.AnthropicContentFields{
		CacheControl: anthropic.CacheControlEphemeralParam{
			Type: constant.ValueOf[constant.Ephemeral](),
			TTL:  ttl,
		},
	}
}

func systemMsg(text string, fields *openai.AnthropicContentFields) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{
		OfSystem: &openai.ChatCompletionSystemMessageParam{
			Role: openai.ChatMessageRoleSystem,
			Content: openai.ContentUnion{Value: []openai.ChatCompletionContentPartTextParam{
				{Type: "text", Text: text, AnthropicContentFields: fields},
			}},
		},
	}
}

func userMsg(text string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{
		OfUser: &openai.ChatCompletionUserMessageParam{
			Role:    openai.ChatMessageRoleUser,
			Content: openai.StringOrUserRoleContentUnion{Value: text},
		},
	}
}

// -----------------------------------------------------------------------
// Fake cachedContents server
// -----------------------------------------------------------------------

type fakeCacheServer struct {
	srv          *httptest.Server
	listResponse string
	createStatus int
	createBody   string
	listCalls    int
	createCalls  int
}

func newFakeCacheServer(t *testing.T, listBody, createBody string, createStatus int) *fakeCacheServer {
	t.Helper()
	f := &fakeCacheServer{
		listResponse: listBody,
		createStatus: createStatus,
		createBody:   createBody,
	}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodGet:
			f.listCalls++
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(f.listResponse))
		case http.MethodPost:
			f.createCalls++
			w.WriteHeader(f.createStatus)
			_, _ = w.Write([]byte(f.createBody))
		}
	}))
	t.Cleanup(f.srv.Close)
	return f
}

// redirectTransport rewrites all outbound requests to hit the fake server host.
type redirectTransport struct {
	fakeHost string
}

func (t *redirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req2 := req.Clone(req.Context())
	req2.URL.Scheme = "http"
	req2.URL.Host = t.fakeHost
	return http.DefaultTransport.RoundTrip(req2)
}

func resolverWithServer(srvURL string) *resolver {
	host := srvURL[len("http://"):]
	return New(&http.Client{
		Transport: &redirectTransport{fakeHost: host},
		Timeout:   5 * time.Second,
	}).(*resolver)
}

// computeKeyForRequest replicates the key generation for use in test assertions.
func computeKeyForRequest(t *testing.T, req *openai.ChatCompletionRequest) string {
	t.Helper()
	bp := findBreakpoint(req.Messages)
	require.GreaterOrEqual(t, bp, 0)
	contents, sys, err := computeKeyInputs(req.Model, req.Messages[:bp+1])
	require.NoError(t, err)
	key, err := computeCacheKey(req.Model, contents, sys, nil)
	require.NoError(t, err)
	return key
}

// -----------------------------------------------------------------------
// Unit tests: findBreakpoint
// -----------------------------------------------------------------------

func TestFindBreakpoint(t *testing.T) {
	tests := []struct {
		name     string
		messages []openai.ChatCompletionMessageParamUnion
		want     int
	}{
		{
			name:     "no markers returns -1",
			messages: []openai.ChatCompletionMessageParamUnion{userMsg("hello")},
			want:     -1,
		},
		{
			name: "system marker at index 0",
			messages: []openai.ChatCompletionMessageParamUnion{
				systemMsg("You are helpful.", ephemeralFields()),
				userMsg("hello"),
			},
			want: 0,
		},
		{
			name: "last marker wins when multiple exist",
			messages: []openai.ChatCompletionMessageParamUnion{
				systemMsg("sys", ephemeralFields()),
				userMsg("q1"),
				{OfUser: &openai.ChatCompletionUserMessageParam{
					Role: openai.ChatMessageRoleUser,
					Content: openai.StringOrUserRoleContentUnion{
						Value: []openai.ChatCompletionContentPartUserUnionParam{
							{OfText: &openai.ChatCompletionContentPartTextParam{
								Type:                   "text",
								Text:                   "q2",
								AnthropicContentFields: ephemeralFields(),
							}},
						},
					},
				}},
				userMsg("q3"),
			},
			want: 2,
		},
		{
			name: "tool message with cache_control",
			messages: []openai.ChatCompletionMessageParamUnion{
				{OfTool: &openai.ChatCompletionToolMessageParam{
					Role:                   openai.ChatMessageRoleTool,
					ToolCallID:             "call_1",
					Content:                openai.ContentUnion{Value: "result"},
					AnthropicContentFields: ephemeralFields(),
				}},
				userMsg("follow-up"),
			},
			want: 0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, findBreakpoint(tc.messages))
		})
	}
}

// -----------------------------------------------------------------------
// Unit tests: extractTTL / anthropicTTLToGCP
// -----------------------------------------------------------------------

func TestExtractTTL(t *testing.T) {
	tests := []struct {
		name    string
		msg     openai.ChatCompletionMessageParamUnion
		wantTTL string
	}{
		{
			name:    "no ttl → default 300s",
			msg:     systemMsg("sys", ephemeralFields()),
			wantTTL: "300s",
		},
		{
			name:    "5m → 300s",
			msg:     systemMsg("sys", ephemeralFieldsWithTTL("5m")),
			wantTTL: "300s",
		},
		{
			name:    "1h → 3600s",
			msg:     systemMsg("sys", ephemeralFieldsWithTTL("1h")),
			wantTTL: "3600s",
		},
		{
			name:    "raw GCP seconds passthrough",
			msg:     systemMsg("sys", ephemeralFieldsWithTTL("600s")),
			wantTTL: "600s",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.wantTTL, extractTTL(tc.msg))
		})
	}
}

// -----------------------------------------------------------------------
// Unit tests: computeCacheKey
// -----------------------------------------------------------------------

func TestComputeCacheKey_Deterministic(t *testing.T) {
	msgs := []openai.ChatCompletionMessageParamUnion{
		systemMsg("You are helpful.", ephemeralFields()),
	}
	contents, sys, err := computeKeyInputs("gemini-1.5-pro", msgs)
	require.NoError(t, err)

	k1, err := computeCacheKey("gemini-1.5-pro", contents, sys, nil)
	require.NoError(t, err)
	k2, err := computeCacheKey("gemini-1.5-pro", contents, sys, nil)
	require.NoError(t, err)
	assert.Equal(t, k1, k2)
	assert.Len(t, k1, 64, "expected SHA-256 hex digest")
}

func TestComputeCacheKey_DifferentModels_DifferentKeys(t *testing.T) {
	msgs := []openai.ChatCompletionMessageParamUnion{systemMsg("sys", ephemeralFields())}
	c, s, err := computeKeyInputs("gemini-1.5-pro", msgs)
	require.NoError(t, err)
	k1, _ := computeCacheKey("gemini-1.5-pro", c, s, nil)
	k2, _ := computeCacheKey("gemini-2.0-flash", c, s, nil)
	assert.NotEqual(t, k1, k2)
}

// -----------------------------------------------------------------------
// Integration tests: Resolve() against fake HTTP server
// -----------------------------------------------------------------------

func TestResolver_NoMarkers_ReturnsNil(t *testing.T) {
	r := New(nil).(*resolver)
	req := &openai.ChatCompletionRequest{
		Model:    "gemini-1.5-pro",
		Messages: []openai.ChatCompletionMessageParamUnion{userMsg("hello")},
	}
	auth := &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}
	res, err := r.Resolve(context.Background(), req, auth)
	require.NoError(t, err)
	assert.Nil(t, res, "no cache_control markers should return nil")
}

func TestResolver_CacheMiss_Creates(t *testing.T) {
	expireISO := time.Now().Add(5 * time.Minute).UTC().Format(time.RFC3339)
	createResp := `{"name":"projects/p/locations/us-central1/cachedContents/new","expireTime":"` + expireISO + `","usageMetadata":{"totalTokenCount":512}}`
	fake := newFakeCacheServer(t, `{"cachedContents":[]}`, createResp, http.StatusOK)
	r := resolverWithServer(fake.srv.URL)

	auth := &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}
	req := &openai.ChatCompletionRequest{
		Model: "gemini-1.5-pro",
		Messages: []openai.ChatCompletionMessageParamUnion{
			systemMsg("You are helpful.", ephemeralFields()),
			userMsg("Hello"),
		},
	}

	res, err := r.Resolve(context.Background(), req, auth)
	require.NoError(t, err)
	require.NotNil(t, res)
	assert.Equal(t, "projects/p/locations/us-central1/cachedContents/new", res.CacheName)
	assert.True(t, res.Created)
	assert.Equal(t, 512, res.TokenCount)
	// remainder = messages after the breakpoint (index 0 → only userMsg at index 1 remains)
	assert.Equal(t, []openai.ChatCompletionMessageParamUnion{userMsg("Hello")}, res.Messages)
	// list → create → re-list (post-create convergence check)
	assert.Equal(t, 2, fake.listCalls)
	assert.Equal(t, 1, fake.createCalls)
}

func TestResolver_CacheHit_FromGoogleList(t *testing.T) {
	req := &openai.ChatCompletionRequest{
		Model: "gemini-1.5-pro",
		Messages: []openai.ChatCompletionMessageParamUnion{
			systemMsg("You are helpful.", ephemeralFields()),
			userMsg("Hello"),
		},
	}
	key := computeKeyForRequest(t, req)
	expireISO := time.Now().Add(5 * time.Minute).UTC().Format(time.RFC3339)

	listBody, err := json.Marshal(map[string]interface{}{
		"cachedContents": []map[string]interface{}{
			{
				"name":        "projects/p/locations/us-central1/cachedContents/existing",
				"displayName": key,
				"model":       "publishers/google/models/gemini-1.5-pro",
				"expireTime":  expireISO,
			},
		},
	})
	require.NoError(t, err)

	fake := newFakeCacheServer(t, string(listBody), "", http.StatusOK)
	r := resolverWithServer(fake.srv.URL)
	auth := &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}

	res, err := r.Resolve(context.Background(), req, auth)
	require.NoError(t, err)
	require.NotNil(t, res)
	assert.Equal(t, "projects/p/locations/us-central1/cachedContents/existing", res.CacheName)
	assert.False(t, res.Created)
	assert.Equal(t, 0, fake.createCalls)
}

func TestResolver_MemoHit_SkipsGoogleAPICalls(t *testing.T) {
	fake := newFakeCacheServer(t, `{"cachedContents":[]}`, "", http.StatusOK)
	r := resolverWithServer(fake.srv.URL)

	req := &openai.ChatCompletionRequest{
		Model: "gemini-1.5-pro",
		Messages: []openai.ChatCompletionMessageParamUnion{
			systemMsg("Cached system.", ephemeralFields()),
			userMsg("q"),
		},
	}
	key := computeKeyForRequest(t, req)
	r.setMemo(key, "projects/p/locations/r/cachedContents/memo-hit", time.Now().Add(10*time.Minute))

	auth := &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}
	res, err := r.Resolve(context.Background(), req, auth)
	require.NoError(t, err)
	require.NotNil(t, res)
	assert.Equal(t, "projects/p/locations/r/cachedContents/memo-hit", res.CacheName)
	assert.False(t, res.Created)
	assert.Equal(t, 0, fake.listCalls, "memo hit must not call the Google API")
	assert.Equal(t, 0, fake.createCalls)
}

func TestResolver_MemoExpiry_Refetches(t *testing.T) {
	req := &openai.ChatCompletionRequest{
		Model: "gemini-1.5-pro",
		Messages: []openai.ChatCompletionMessageParamUnion{
			systemMsg("sys", ephemeralFields()),
			userMsg("q"),
		},
	}
	key := computeKeyForRequest(t, req)
	expireISO := time.Now().Add(5 * time.Minute).UTC().Format(time.RFC3339)

	listBody, _ := json.Marshal(map[string]interface{}{
		"cachedContents": []map[string]interface{}{
			{
				"name":        "projects/p/locations/us-central1/cachedContents/refreshed",
				"displayName": key,
				"model":       "publishers/google/models/gemini-1.5-pro",
				"expireTime":  expireISO,
			},
		},
	})
	fake := newFakeCacheServer(t, string(listBody), "", http.StatusOK)
	r := resolverWithServer(fake.srv.URL)
	// Seed with entry expiring within the 10 s stale window → should be evicted.
	r.setMemo(key, "projects/p/locations/us-central1/cachedContents/stale", time.Now().Add(5*time.Second))

	auth := &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}
	res, err := r.Resolve(context.Background(), req, auth)
	require.NoError(t, err)
	require.NotNil(t, res)
	assert.Equal(t, "projects/p/locations/us-central1/cachedContents/refreshed", res.CacheName)
	assert.Equal(t, 1, fake.listCalls, "stale memo must trigger a list call")
}

func TestResolver_CreateFailure_ReturnsError(t *testing.T) {
	fake := newFakeCacheServer(t,
		`{"cachedContents":[]}`,
		`{"error":{"message":"below minimum tokens"}}`,
		http.StatusUnprocessableEntity,
	)
	r := resolverWithServer(fake.srv.URL)
	auth := &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}
	req := &openai.ChatCompletionRequest{
		Model: "gemini-1.5-pro",
		Messages: []openai.ChatCompletionMessageParamUnion{
			systemMsg("sys", ephemeralFields()),
			userMsg("q"),
		},
	}
	_, err := r.Resolve(context.Background(), req, auth)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "422")
}

func TestResolver_TTLDefault_SentToGoogle(t *testing.T) {
	expireISO := time.Now().Add(5 * time.Minute).UTC().Format(time.RFC3339)
	var capturedBody []byte

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"cachedContents":[]}`))
			return
		}
		buf := make([]byte, 8192)
		n, _ := r.Body.Read(buf)
		capturedBody = buf[:n]
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"name":"projects/p/locations/r/cachedContents/x","expireTime":"` + expireISO + `"}`))
	}))
	defer srv.Close()

	r := resolverWithServer(srv.URL)
	auth := &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}
	req := &openai.ChatCompletionRequest{
		Model: "gemini-1.5-pro",
		Messages: []openai.ChatCompletionMessageParamUnion{
			systemMsg("sys", ephemeralFields()), // no TTL → default 300s
			userMsg("q"),
		},
	}
	_, err := r.Resolve(context.Background(), req, auth)
	require.NoError(t, err)

	var body createCacheRequest
	require.NoError(t, json.Unmarshal(capturedBody, &body))
	assert.Equal(t, defaultTTL, body.TTL)
}

func TestResolver_TTLOverride_SentToGoogle(t *testing.T) {
	expireISO := time.Now().Add(time.Hour).UTC().Format(time.RFC3339)
	var capturedBody []byte

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"cachedContents":[]}`))
			return
		}
		buf := make([]byte, 8192)
		n, _ := r.Body.Read(buf)
		capturedBody = buf[:n]
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"name":"projects/p/locations/r/cachedContents/x","expireTime":"` + expireISO + `"}`))
	}))
	defer srv.Close()

	r := resolverWithServer(srv.URL)
	auth := &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}
	req := &openai.ChatCompletionRequest{
		Model: "gemini-1.5-pro",
		Messages: []openai.ChatCompletionMessageParamUnion{
			systemMsg("sys", ephemeralFieldsWithTTL("1h")),
			userMsg("q"),
		},
	}
	_, err := r.Resolve(context.Background(), req, auth)
	require.NoError(t, err)

	var body createCacheRequest
	require.NoError(t, json.Unmarshal(capturedBody, &body))
	assert.Equal(t, "3600s", body.TTL)
}

func TestResolver_DuplicateCreateRace_UsesListResult(t *testing.T) {
	req := &openai.ChatCompletionRequest{
		Model: "gemini-1.5-pro",
		Messages: []openai.ChatCompletionMessageParamUnion{
			systemMsg("sys", ephemeralFields()),
			userMsg("q"),
		},
	}
	key := computeKeyForRequest(t, req)
	expireISO := time.Now().Add(5 * time.Minute).UTC().Format(time.RFC3339)

	listCalls := 0
	// First list: empty (cache not yet created by this replica).
	// Create: returns "new-by-us".
	// Second list (post-create): returns "older-by-other" — simulates another replica winning the race.
	listBodyEmpty := `{"cachedContents":[]}`
	listBodyOther, _ := json.Marshal(map[string]interface{}{
		"cachedContents": []map[string]interface{}{
			{
				"name":        "projects/p/locations/us-central1/cachedContents/older-by-other",
				"displayName": key,
				"model":       "publishers/google/models/gemini-1.5-pro",
				"expireTime":  expireISO,
			},
		},
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodGet:
			listCalls++
			w.WriteHeader(http.StatusOK)
			if listCalls == 1 {
				_, _ = w.Write([]byte(listBodyEmpty))
			} else {
				_, _ = w.Write(listBodyOther)
			}
		case http.MethodPost:
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"name":"projects/p/locations/us-central1/cachedContents/new-by-us","expireTime":"` + expireISO + `"}`))
		}
	}))
	defer srv.Close()

	r := resolverWithServer(srv.URL)
	auth := &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}
	res, err := r.Resolve(context.Background(), req, auth)
	require.NoError(t, err)
	require.NotNil(t, res)
	// Resolver should prefer the entry from the post-create re-list (the other replica's entry).
	assert.Equal(t, "projects/p/locations/us-central1/cachedContents/older-by-other", res.CacheName)
	// Created should be false when we converged to another replica's cache.
	assert.False(t, res.Created)
}
