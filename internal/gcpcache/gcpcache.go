// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

// Package gcpcache implements context-cache resolution for GCP Vertex AI (Gemini).
//
// When a client sends an OpenAI-format request with Anthropic-style cache_control markers,
// the resolver:
//  1. Finds the last cache_control breakpoint in the message list.
//  2. Splits the request into a cached prefix (tools + system + messages up to the breakpoint)
//     and a non-cached remainder.
//  3. Generates a deterministic SHA-256 cache key and looks it up in the shared store.
//  4. On a store miss, lists Google cachedContents for the target region and model.
//  5. Creates the cache entry if not found, then returns the cache resource name and the
//     non-cached messages.
//
// Concurrent resolutions of the same key are deduplicated by the store's create lock,
// which coordinates across gateway replicas rather than only within one process.
//
// The CacheResolver interface allows the implementation to be replaced with an external
// cache service in the future without changing the request-path callers.
package gcpcache

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"google.golang.org/genai"

	"github.com/envoyproxy/ai-gateway/internal/apischema/gcp"
	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/filterapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
	"github.com/envoyproxy/ai-gateway/internal/translator"
)

const (
	// defaultTTL is the default cache TTL when none is specified in the cache_control marker.
	defaultTTL = "300s"

	// gcpCachedContentsBasePath is the base URL for the Vertex AI cachedContents REST API.
	gcpCachedContentsBasePath = "https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/cachedContents"

	// staleThreshold is how close to its expiry a stored entry must be before it is
	// treated as a miss, so there is time to re-resolve before the Google cache
	// disappears underneath an in-flight request.
	staleThreshold = 10 * time.Second
)

// ResolveResult holds the outcome of a successful cache resolution.
type ResolveResult struct {
	// CacheName is the full Google resource name of the resolved or created cache entry.
	// Format: "projects/{project}/locations/{location}/cachedContents/{cache_id}"
	CacheName string
	// Messages is the non-cached remainder of the conversation (messages after the breakpoint).
	Messages []openai.ChatCompletionMessageParamUnion
	// Created is true when this call created a new cache entry (cache-write cost applies).
	Created bool
	// TokenCount is the number of tokens stored in the cache (from Google's create response).
	// Only populated when Created is true.
	TokenCount int
	// ExpireTime is the cache expiration time reported by Google.
	ExpireTime time.Time
}

// CacheResolver resolves or creates a GCP Vertex AI cached content entry for requests
// that carry Anthropic-style cache_control markers.
type CacheResolver interface {
	// Resolve inspects openAIReq for cache_control markers, resolves or creates the
	// corresponding Google cachedContents entry, and returns the result.
	// The caller is responsible for injecting ResolveResult.CacheName as the
	// cachedContent field on the Gemini request and replacing the request messages
	// with ResolveResult.Messages.
	Resolve(ctx context.Context, openAIReq *openai.ChatCompletionRequest, gcpAuth filterapi.GCPAuthHandler) (*ResolveResult, error)
}

// resolver is the default CacheResolver implementation.
type resolver struct {
	httpClient *http.Client

	// store holds resolved cache names. It is shared across replicas when backed by
	// Redis, which is what keeps two replicas from creating the same cache. Its
	// failures are never fatal: they are logged and treated as a miss.
	store CacheStore

	// logger records store failures, which are otherwise invisible because the request
	// proceeds normally.
	logger *slog.Logger
}

// New creates a new CacheResolver.
//
// httpClient is used for calls to the Google cachedContents API; pass nil for a default.
// store holds resolved cache names; pass nil to disable caching entirely, which makes
// resolution inert rather than failing requests. logger may be nil.
func New(httpClient *http.Client, store CacheStore, logger *slog.Logger) CacheResolver {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 30 * time.Second}
	}
	if store == nil {
		store = noopStore{}
	}
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	return &resolver{httpClient: httpClient, store: store, logger: logger}
}

// Resolve implements CacheResolver.
func (r *resolver) Resolve(ctx context.Context, openAIReq *openai.ChatCompletionRequest, gcpAuth filterapi.GCPAuthHandler) (*ResolveResult, error) {
	// Find the last cache_control breakpoint in the message list.
	breakpoint := findBreakpoint(openAIReq.Messages)
	if breakpoint < 0 {
		// No markers — nothing to cache.
		return nil, nil
	}

	// Split cached prefix.
	cachedMessages := openAIReq.Messages[:breakpoint+1]
	remainderMessages := openAIReq.Messages[breakpoint+1:]

	// Extract the TTL from the breakpoint message. If not found, fall back to default.
	ttl := extractTTL(openAIReq.Messages[breakpoint])

	// Build the Gemini cached prefix (tools + system + contents).
	contents, systemInstruction, err := translator.OpenAIMessagesToGeminiContents(cachedMessages, openAIReq.Model)
	if err != nil {
		return nil, fmt.Errorf("gcpcache: failed to convert cached messages to Gemini format: %w", err)
	}
	geminiTools, err := translator.OpenAIToolsToGeminiTools(openAIReq.Tools, false)
	if err != nil {
		return nil, fmt.Errorf("gcpcache: failed to convert tools to Gemini format: %w", err)
	}

	// Compute a deterministic cache key.
	cacheKey, err := computeCacheKey(openAIReq.Model, contents, systemInstruction, geminiTools)
	if err != nil {
		return nil, fmt.Errorf("gcpcache: failed to compute cache key: %w", err)
	}

	// Check the shared store first. A store failure is not fatal: it is logged and
	// treated as a miss, so an unreachable store degrades caching rather than the
	// request. Google-side failures below are a different matter and do fail fast.
	if e, ok := r.storeGet(ctx, cacheKey); ok {
		return &ResolveResult{
			CacheName:  e.cacheName,
			Messages:   remainderMessages,
			ExpireTime: e.expireTime,
		}, nil
	}

	// Store miss — resolve against GCP. Concurrent resolutions of the same key are
	// arbitrated by the store's create lock (see resolveUncached), which deduplicates
	// across replicas rather than only within this one.
	result, err := r.resolveUncached(ctx, openAIReq, gcpAuth, cacheKey, ttl, contents, systemInstruction, geminiTools)
	if err != nil {
		return nil, err
	}

	// resolveUncached leaves Messages unset because it deals only in the cached prefix;
	// the remainder is per-request.
	result.Messages = remainderMessages
	return result, nil
}

// resolveUncached performs the GCP-side resolution for a cache key: list, create if
// absent, then re-list to converge duplicate-create races. It deals only in the cached
// prefix, so it leaves ResolveResult.Messages unset for the caller to fill in per-request.
//
// Before creating, it claims a store-level lock so that replicas racing on a cold prefix
// produce one create rather than N. Losing the race means waiting for the winner's result;
// the wait degrading into a plain resolution is safe, because the list below is the actual
// source of truth.
func (r *resolver) resolveUncached(
	ctx context.Context,
	openAIReq *openai.ChatCompletionRequest,
	gcpAuth filterapi.GCPAuthHandler,
	cacheKey, ttl string,
	contents []genai.Content,
	systemInstruction *genai.Content,
	geminiTools []genai.Tool,
) (*ResolveResult, error) {
	region := gcpAuth.GCPRegion()
	project := gcpAuth.GCPProject()

	// Get access token.
	tokenSrc := gcpAuth.GCPTokenSource()
	token, err := tokenSrc.Token()
	if err != nil {
		return nil, fmt.Errorf("gcpcache: failed to get GCP access token: %w", err)
	}
	accessToken := token.AccessToken

	// Claim the right to create this key. A replica that loses waits for the winner's
	// result; if the winner never publishes one, it falls through and resolves itself.
	locked := r.tryLock(ctx, cacheKey)
	if !locked {
		if e, ok := r.awaitLeader(ctx, cacheKey); ok {
			return &ResolveResult{CacheName: e.cacheName, ExpireTime: e.expireTime}, nil
		}
	}

	// List existing caches and match by displayName (cacheKey).
	baseURL := fmt.Sprintf(gcpCachedContentsBasePath, region, project, region)
	existingName, expireTime, err := r.listAndMatch(ctx, baseURL, accessToken, cacheKey, openAIReq.Model)
	if err != nil {
		if locked {
			r.unlock(ctx, cacheKey)
		}
		return nil, fmt.Errorf("gcpcache: failed to list cached contents: %w", err)
	}

	if existingName != "" {
		r.storeSet(ctx, cacheKey, entry{cacheName: existingName, expireTime: expireTime})
		return &ResolveResult{
			CacheName:  existingName,
			ExpireTime: expireTime,
		}, nil
	}

	// Cache not found — create it. A replica that did not win the lock still creates
	// here, having already waited for the winner without result; this is the duplicate
	// the lock narrows but cannot fully eliminate.
	created, tokenCount, expireTime, err := r.createCache(ctx, baseURL, accessToken, openAIReq.Model, region, project, cacheKey, contents, systemInstruction, geminiTools, ttl)
	if err != nil {
		// Release the claim so waiters retry immediately rather than blocking for the
		// remainder of the lock TTL on a create that will never publish.
		if locked {
			r.unlock(ctx, cacheKey)
		}
		return nil, fmt.Errorf("gcpcache: failed to create cached content: %w", err)
	}

	// After create, re-list and prefer the oldest match to converge duplicate-create races.
	finalName, finalExpire, listErr := r.listAndMatch(ctx, baseURL, accessToken, cacheKey, openAIReq.Model)
	if listErr == nil && finalName != "" && finalName != created {
		// Another replica created a cache with the same key; use the one from the list.
		r.storeSet(ctx, cacheKey, entry{cacheName: finalName, expireTime: finalExpire})
		return &ResolveResult{
			CacheName:  finalName,
			ExpireTime: finalExpire,
		}, nil
	}

	r.storeSet(ctx, cacheKey, entry{cacheName: created, expireTime: expireTime})
	return &ResolveResult{
		CacheName:  created,
		Created:    true,
		TokenCount: tokenCount,
		ExpireTime: expireTime,
	}, nil
}

// -----------------------------------------------------------------------
// Store access
//
// The resolver never propagates a store failure to its caller: caching is an
// optimization, and an unreachable store must not turn a servable request into an
// error. Failures are logged and treated as a miss, leaving Google as the source of
// truth. Google-side failures are propagated, because those are actionable by the
// user — a create rejected for being below the model's minimum token count means the
// cache_control markers are misplaced, and silently serving the request uncached
// would hide a cost increase.
// -----------------------------------------------------------------------

// storeGet reads a resolved cache name, reporting a miss on any failure.
func (r *resolver) storeGet(ctx context.Context, key string) (entry, bool) {
	e, ok, err := r.store.Get(ctx, key)
	if err != nil {
		r.logger.Warn("gcpcache: cache store read failed, proceeding uncached",
			slog.String("error", err.Error()))
		return entry{}, false
	}
	if !ok {
		return entry{}, false
	}
	// Treat entries expiring within 10s as stale so there is time to re-resolve before
	// the cache disappears underneath an in-flight request.
	if time.Until(e.expireTime) < staleThreshold {
		return entry{}, false
	}
	return e, true
}

// storeSet records a resolved cache name, expiring it with the Google entry itself so
// the store cannot outlive what it points at.
func (r *resolver) storeSet(ctx context.Context, key string, e entry) {
	ttl := time.Until(e.expireTime)
	if ttl <= 0 {
		return
	}
	if err := r.store.Set(ctx, key, e, ttl); err != nil {
		r.logger.Warn("gcpcache: cache store write failed",
			slog.String("error", err.Error()))
	}
}

// locker is implemented by stores that can arbitrate cache creation across replicas.
// Stores that cannot (such as the no-op store) simply never grant a lock, which leaves
// every caller resolving independently — correct, just without the deduplication.
type locker interface {
	tryLock(ctx context.Context, key string) (bool, error)
	unlock(ctx context.Context, key string)
	awaitLeader(ctx context.Context, key string) (entry, bool, error)
}

// tryLock claims the right to create key. It reports false when the store cannot lock,
// when the lock is held elsewhere, or on failure — all cases where the caller should
// resolve against Google rather than wait.
func (r *resolver) tryLock(ctx context.Context, key string) bool {
	l, ok := r.store.(locker)
	if !ok {
		return false
	}
	won, err := l.tryLock(ctx, key)
	if err != nil {
		r.logger.Warn("gcpcache: cache store lock failed, proceeding without coordination",
			slog.String("error", err.Error()))
		return false
	}
	return won
}

func (r *resolver) unlock(ctx context.Context, key string) {
	if l, ok := r.store.(locker); ok {
		l.unlock(ctx, key)
	}
}

// awaitLeader waits for the replica holding the lock to publish its result.
func (r *resolver) awaitLeader(ctx context.Context, key string) (entry, bool) {
	l, ok := r.store.(locker)
	if !ok {
		return entry{}, false
	}
	e, found, err := l.awaitLeader(ctx, key)
	if err != nil {
		// Includes the timeout case: fall through and resolve rather than fail.
		return entry{}, false
	}
	return e, found
}

// -----------------------------------------------------------------------
// Breakpoint and TTL helpers
// -----------------------------------------------------------------------

// findBreakpoint returns the index of the last message that contains a
// cache_control marker, or -1 if none is found.
func findBreakpoint(messages []openai.ChatCompletionMessageParamUnion) int {
	for i := len(messages) - 1; i >= 0; i-- {
		if messageHasCacheControl(&messages[i]) {
			return i
		}
	}
	return -1
}

// messageHasCacheControl returns true if the message contains at least one
// content part with an Anthropic-style cache_control marker.
func messageHasCacheControl(msg *openai.ChatCompletionMessageParamUnion) bool {
	if msg.OfTool != nil && isCacheControlSet(msg.OfTool.AnthropicContentFields) {
		return true
	}
	if msg.OfSystem != nil {
		if parts, ok := msg.OfSystem.Content.Value.([]openai.ChatCompletionContentPartTextParam); ok {
			for i := range parts {
				if isCacheControlSet(parts[i].AnthropicContentFields) {
					return true
				}
			}
		}
	}
	if msg.OfUser != nil {
		if parts, ok := msg.OfUser.Content.Value.([]openai.ChatCompletionContentPartUserUnionParam); ok {
			for i := range parts {
				p := &parts[i]
				if p.OfText != nil && isCacheControlSet(p.OfText.AnthropicContentFields) {
					return true
				}
				if p.OfImageURL != nil && isCacheControlSet(p.OfImageURL.AnthropicContentFields) {
					return true
				}
				if p.OfInputAudio != nil && isCacheControlSet(p.OfInputAudio.AnthropicContentFields) {
					return true
				}
			}
		}
	}
	return false
}

// isCacheControlSet returns true if the AnthropicContentFields carries an ephemeral cache marker.
func isCacheControlSet(fields *openai.AnthropicContentFields) bool {
	return fields != nil && string(fields.CacheControl.Type) == "ephemeral"
}

// extractTTL returns the GCP TTL string (e.g. "3600s") from the last cache_control
// marker found in the message. Falls back to defaultTTL when none is specified.
// Anthropic TTL values ("5m", "1h") are converted to GCP seconds format.
func extractTTL(msg openai.ChatCompletionMessageParamUnion) string {
	var ttl string
	checkFields := func(f *openai.AnthropicContentFields) {
		if f != nil && string(f.CacheControl.Type) == "ephemeral" && string(f.CacheControl.TTL) != "" {
			ttl = anthropicTTLToGCP(string(f.CacheControl.TTL))
		}
	}
	if msg.OfTool != nil {
		checkFields(msg.OfTool.AnthropicContentFields)
	}
	if msg.OfSystem != nil {
		if parts, ok := msg.OfSystem.Content.Value.([]openai.ChatCompletionContentPartTextParam); ok {
			for i := range parts {
				checkFields(parts[i].AnthropicContentFields)
			}
		}
	}
	if msg.OfUser != nil {
		if parts, ok := msg.OfUser.Content.Value.([]openai.ChatCompletionContentPartUserUnionParam); ok {
			for i := range parts {
				p := &parts[i]
				if p.OfText != nil {
					checkFields(p.OfText.AnthropicContentFields)
				}
				if p.OfImageURL != nil {
					checkFields(p.OfImageURL.AnthropicContentFields)
				}
				if p.OfInputAudio != nil {
					checkFields(p.OfInputAudio.AnthropicContentFields)
				}
			}
		}
	}
	if ttl == "" {
		return defaultTTL
	}
	return ttl
}

// anthropicTTLToGCP converts Anthropic cache TTL values to GCP seconds format.
// Anthropic supports "5m" and "1h"; unknown values fall back to defaultTTL.
func anthropicTTLToGCP(ttl string) string {
	switch ttl {
	case "5m":
		return "300s"
	case "1h":
		return "3600s"
	default:
		// If the value is already in GCP seconds format (e.g. "600s"), pass it through.
		if len(ttl) > 1 && ttl[len(ttl)-1] == 's' {
			return ttl
		}
		return defaultTTL
	}
}

// computeKeyInputs converts a cached message prefix into Gemini format for key generation.
func computeKeyInputs(model string, cachedMessages []openai.ChatCompletionMessageParamUnion) ([]genai.Content, *genai.Content, error) {
	contents, sys, err := translator.OpenAIMessagesToGeminiContents(cachedMessages, model)
	if err != nil {
		return nil, nil, err
	}
	return contents, sys, nil
}

// -----------------------------------------------------------------------
// Cache key
// -----------------------------------------------------------------------

type cacheKeyInput struct {
	Model             string          `json:"model"`
	Contents          []genai.Content `json:"contents"`
	SystemInstruction *genai.Content  `json:"systemInstruction,omitempty"`
	Tools             []genai.Tool    `json:"tools,omitempty"`
}

// computeCacheKey generates a deterministic SHA-256 hex digest over the
// (model, contents, systemInstruction, tools) tuple. The digest is used as the
// Google cachedContents displayName.
func computeCacheKey(model string, contents []genai.Content, systemInstruction *genai.Content, tools []genai.Tool) (string, error) {
	input := cacheKeyInput{
		Model:             model,
		Contents:          contents,
		SystemInstruction: systemInstruction,
		Tools:             tools,
	}
	b, err := json.Marshal(input)
	if err != nil {
		return "", fmt.Errorf("failed to marshal cache key input: %w", err)
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), nil
}

// -----------------------------------------------------------------------
// Google cachedContents REST API calls
// -----------------------------------------------------------------------

// cachedContentItem is the subset of the cachedContents list item we care about.
type cachedContentItem struct {
	Name        string `json:"name"`
	DisplayName string `json:"displayName"`
	ExpireTime  string `json:"expireTime"` // RFC 3339
	Model       string `json:"model"`
}

type listResponse struct {
	CachedContents []cachedContentItem `json:"cachedContents"`
}

// listAndMatch lists cachedContents for the given region+project and returns the
// name of the first entry whose displayName matches cacheKey and whose model
// suffix matches the requested model. Returns ("", zero, nil) when not found.
func (r *resolver) listAndMatch(ctx context.Context, baseURL, accessToken, cacheKey, model string) (string, time.Time, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL, nil)
	if err != nil {
		return "", time.Time{}, err
	}
	req.Header.Set("Authorization", "Bearer "+accessToken)

	resp, err := r.httpClient.Do(req)
	if err != nil {
		return "", time.Time{}, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", time.Time{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return "", time.Time{}, fmt.Errorf("list cachedContents returned HTTP %d: %s", resp.StatusCode, body)
	}

	var lr listResponse
	if err = json.Unmarshal(body, &lr); err != nil {
		return "", time.Time{}, fmt.Errorf("failed to decode list response: %w", err)
	}

	for _, item := range lr.CachedContents {
		if item.DisplayName != cacheKey {
			continue
		}
		// Model in the item is the full resource name; check the suffix.
		if !modelMatchesSuffix(item.Model, model) {
			continue
		}
		expireTime, _ := time.Parse(time.RFC3339, item.ExpireTime)
		return item.Name, expireTime, nil
	}
	return "", time.Time{}, nil
}

// modelMatchesSuffix checks whether the full model resource name ends with the
// requested short model name (e.g. "publishers/google/models/gemini-1.5-pro").
func modelMatchesSuffix(fullModel, shortModel string) bool {
	if fullModel == shortModel {
		return true
	}
	suffix := "models/" + shortModel
	return len(fullModel) >= len(suffix) && fullModel[len(fullModel)-len(suffix):] == suffix
}

// createCache creates a cached content in GCP and returns the new cache's resource name,
// token count, and expiry time.
func (r *resolver) createCache(
	ctx context.Context,
	baseURL, accessToken, model, region, project, cacheKey string,
	contents []genai.Content,
	systemInstruction *genai.Content,
	tools []genai.Tool,
	ttl string,
) (name string, tokenCount int, expireTime time.Time, err error) {
	// Vertex AI expects the full model resource name.
	fullModel := fmt.Sprintf("projects/%s/locations/%s/publishers/google/models/%s", project, region, model)

	body := gcp.CreateCachedContent{
		Model:             fullModel,
		Contents:          contents,
		SystemInstruction: systemInstruction,
		Tools:             tools,
		DisplayName:       cacheKey,
		TTL:               ttl,
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return "", 0, time.Time{}, fmt.Errorf("failed to marshal create cache request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return "", 0, time.Time{}, err
	}
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.httpClient.Do(req)
	if err != nil {
		return "", 0, time.Time{}, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, time.Time{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return "", 0, time.Time{}, fmt.Errorf("create cachedContent returned HTTP %d: %s", resp.StatusCode, respBody)
	}

	var cr gcp.CachedContent
	if err = json.Unmarshal(respBody, &cr); err != nil {
		return "", 0, time.Time{}, fmt.Errorf("failed to decode create cache response: %w", err)
	}

	if cr.UsageMetadata != nil {
		tokenCount = int(cr.UsageMetadata.TotalTokenCount)
	}
	return cr.Name, tokenCount, cr.ExpireTime, nil
}
