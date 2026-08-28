// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package gcpcache

import (
	"context"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/cachestore"
)

// -----------------------------------------------------------------------
// Cross-replica behavior
// -----------------------------------------------------------------------

// Two resolvers stand in for two gateway replicas: separate processes, separate
// singleflight groups, one shared Redis. This is the case the in-memory store could not
// cover, and the reason the store exists.
func TestResolver_CrossReplica_SingleCreate(t *testing.T) {
	expireISO := time.Now().Add(5 * time.Minute).UTC().Format(time.RFC3339)
	createResp := `{"name":"projects/p/locations/us-central1/cachedContents/new","expireTime":"` + expireISO + `","usageMetadata":{"totalTokenCount":512}}`
	fake := newFakeCacheServer(t, `{"cachedContents":[]}`, createResp, http.StatusOK)

	mr := miniredis.RunT(t)
	newReplica := func() *resolver {
		store, err := cachestore.NewRedisStore(mr.Addr())
		require.NoError(t, err)
		return resolverWithServer(fake.srv.URL, store)
	}
	a, b := newReplica(), newReplica()

	req := func() *openai.ChatCompletionRequest {
		return &openai.ChatCompletionRequest{
			Model: "gemini-1.5-pro",
			Messages: []openai.ChatCompletionMessageParamUnion{
				systemMsg("You are helpful.", ephemeralFields()),
				userMsg("Hello"),
			},
		}
	}
	auth := &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}

	var start, done sync.WaitGroup
	start.Add(1)
	results := make([]*ResolveResult, 2)
	errs := make([]error, 2)
	for i, r := range []*resolver{a, b} {
		done.Add(1)
		go func() {
			defer done.Done()
			start.Wait()
			results[i], errs[i] = r.Resolve(context.Background(), req(), auth)
		}()
	}
	start.Done()
	done.Wait()

	for i := range results {
		require.NoError(t, errs[i], "replica %d", i)
		require.NotNil(t, results[i], "replica %d", i)
		assert.Equal(t, "projects/p/locations/us-central1/cachedContents/new", results[i].CacheName)
	}
	assert.Equal(t, 1, fake.creates(), "two replicas sharing one Redis must create exactly one cache")

	// Only the replica that performed the write may bill for it.
	created := 0
	for _, res := range results {
		if res.Created {
			created++
		}
	}
	assert.Equal(t, 1, created, "exactly one replica may report Created")
}

// The failure policy: an unreachable store degrades caching, it does not fail requests.
// The resolver falls back to Google and the request is served normally.
func TestResolver_StoreUnreachable_FailsOpen(t *testing.T) {
	expireISO := time.Now().Add(5 * time.Minute).UTC().Format(time.RFC3339)
	createResp := `{"name":"projects/p/locations/us-central1/cachedContents/new","expireTime":"` + expireISO + `","usageMetadata":{"totalTokenCount":512}}`
	fake := newFakeCacheServer(t, `{"cachedContents":[]}`, createResp, http.StatusOK)

	mr := miniredis.RunT(t)
	store, err := cachestore.NewRedisStore(mr.Addr())
	require.NoError(t, err)
	mr.Close() // Every operation now fails to dial.

	r := resolverWithServer(fake.srv.URL, store)
	req := &openai.ChatCompletionRequest{
		Model: "gemini-1.5-pro",
		Messages: []openai.ChatCompletionMessageParamUnion{
			systemMsg("You are helpful.", ephemeralFields()),
			userMsg("Hello"),
		},
	}
	auth := &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}

	res, err := r.Resolve(context.Background(), req, auth)
	require.NoError(t, err, "a dead store must not fail the request")
	require.NotNil(t, res)
	assert.Equal(t, "projects/p/locations/us-central1/cachedContents/new", res.CacheName)
	assert.Equal(t, 1, fake.creates(), "the resolver falls through to Google")
}
