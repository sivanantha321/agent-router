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
)

// newTestRedisStore starts an in-process Redis and returns a store pointed at it.
func newTestRedisStore(t *testing.T) (*redisStore, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.RunT(t)
	s, err := NewRedisStore(mr.Addr())
	require.NoError(t, err)
	return s.(*redisStore), mr
}

func TestParseRedisURL(t *testing.T) {
	t.Run("bare host:port", func(t *testing.T) {
		opts, err := parseRedisURL("localhost:6379")
		require.NoError(t, err)
		assert.Equal(t, "localhost:6379", opts.Addr)
	})
	t.Run("scheme-qualified", func(t *testing.T) {
		opts, err := parseRedisURL("redis://user:pw@localhost:6380/2")
		require.NoError(t, err)
		assert.Equal(t, "localhost:6380", opts.Addr)
		assert.Equal(t, 2, opts.DB)
	})
	t.Run("empty", func(t *testing.T) {
		_, err := parseRedisURL("")
		require.Error(t, err)
	})
	t.Run("malformed", func(t *testing.T) {
		_, err := parseRedisURL("http://localhost:6379")
		require.Error(t, err)
	})
}

func TestRedisStore_SetGetRoundTrip(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx := context.Background()
	expire := time.Now().Add(10 * time.Minute).UTC().Truncate(time.Second)

	require.NoError(t, s.Set(ctx, "k", entry{cacheName: "projects/p/locations/r/cachedContents/x", expireTime: expire}, time.Minute))

	got, ok, err := s.Get(ctx, "k")
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, "projects/p/locations/r/cachedContents/x", got.cacheName)
	assert.True(t, expire.Equal(got.expireTime), "want %s got %s", expire, got.expireTime)
}

func TestRedisStore_GetMiss(t *testing.T) {
	s, _ := newTestRedisStore(t)
	_, ok, err := s.Get(context.Background(), "absent")
	require.NoError(t, err)
	assert.False(t, ok)
}

// A key holding the create sentinel is a claim, not a result: reporting it as a hit would
// hand the caller the sentinel string as a cache name.
func TestRedisStore_SentinelIsAMiss(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx := context.Background()

	won, err := s.tryLock(ctx, "k")
	require.NoError(t, err)
	require.True(t, won)

	_, ok, err := s.Get(ctx, "k")
	require.NoError(t, err)
	assert.False(t, ok, "an in-flight create must not read as a cache hit")
}

// A value that does not decode is treated as a miss rather than an error: it is not worth
// failing a resolution over, and the next write overwrites it.
func TestRedisStore_MalformedValueIsAMiss(t *testing.T) {
	s, mr := newTestRedisStore(t)
	require.NoError(t, mr.Set("k", "not-an-entry"))

	_, ok, err := s.Get(context.Background(), "k")
	require.NoError(t, err)
	assert.False(t, ok)
}

func TestRedisStore_TryLockIsExclusive(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx := context.Background()

	first, err := s.tryLock(ctx, "k")
	require.NoError(t, err)
	assert.True(t, first)

	second, err := s.tryLock(ctx, "k")
	require.NoError(t, err)
	assert.False(t, second, "a second claim on a held key must lose")
}

// unlock must not delete a key that has moved on from the sentinel, or it would erase a
// published cache name (or another replica's fresh claim after the lock expired).
func TestRedisStore_UnlockOnlyRemovesOwnSentinel(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx := context.Background()
	expire := time.Now().Add(10 * time.Minute)

	won, err := s.tryLock(ctx, "k")
	require.NoError(t, err)
	require.True(t, won)
	require.NoError(t, s.Set(ctx, "k", entry{cacheName: "published", expireTime: expire}, time.Minute))

	s.unlock(ctx, "k")

	got, ok, err := s.Get(ctx, "k")
	require.NoError(t, err)
	require.True(t, ok, "unlock must not erase a published result")
	assert.Equal(t, "published", got.cacheName)
}

func TestRedisStore_UnlockReleasesClaim(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx := context.Background()

	won, err := s.tryLock(ctx, "k")
	require.NoError(t, err)
	require.True(t, won)
	s.unlock(ctx, "k")

	again, err := s.tryLock(ctx, "k")
	require.NoError(t, err)
	assert.True(t, again, "a released claim must be re-claimable immediately")
}

func TestRedisStore_AwaitLeader_ReturnsPublishedResult(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx := context.Background()
	expire := time.Now().Add(10 * time.Minute).UTC().Truncate(time.Second)

	won, err := s.tryLock(ctx, "k")
	require.NoError(t, err)
	require.True(t, won)

	go func() {
		time.Sleep(100 * time.Millisecond)
		_ = s.Set(ctx, "k", entry{cacheName: "from-leader", expireTime: expire}, time.Minute)
	}()

	got, ok, err := s.awaitLeader(ctx, "k")
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, "from-leader", got.cacheName)
}

// A leader that releases without publishing must not strand its waiters for the rest of
// lockWaitTime; they are told to resolve themselves as soon as the claim disappears.
func TestRedisStore_AwaitLeader_ClaimReleasedWithoutResult(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx := context.Background()

	won, err := s.tryLock(ctx, "k")
	require.NoError(t, err)
	require.True(t, won)

	go func() {
		time.Sleep(100 * time.Millisecond)
		s.unlock(ctx, "k")
	}()

	start := time.Now()
	_, ok, err := s.awaitLeader(ctx, "k")
	require.ErrorIs(t, err, errLockHeld)
	assert.False(t, ok)
	assert.Less(t, time.Since(start), lockWaitTime, "waiter must not block for the full wait window")
}

func TestRedisStore_AwaitLeader_HonorsContextCancellation(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx, cancel := context.WithCancel(context.Background())

	won, err := s.tryLock(ctx, "k")
	require.NoError(t, err)
	require.True(t, won)

	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	_, _, err = s.awaitLeader(ctx, "k")
	require.ErrorIs(t, err, context.Canceled)
}

func TestEncodeDecodeEntry(t *testing.T) {
	expire := time.Now().Add(time.Hour).UTC().Truncate(time.Second)
	// The cache name is encoded last precisely because it may contain the separator.
	e := entry{cacheName: "projects/p|weird/cachedContents/x", expireTime: expire}

	got, err := decodeEntry(encodeEntry(e))
	require.NoError(t, err)
	assert.Equal(t, e.cacheName, got.cacheName)
	assert.True(t, expire.Equal(got.expireTime))

	for _, bad := range []string{"", "no-separator", "not-a-time|name", "2020-01-01T00:00:00Z|"} {
		_, err := decodeEntry(bad)
		assert.Error(t, err, "input %q", bad)
	}
}

// -----------------------------------------------------------------------
// Cross-replica behavior
// -----------------------------------------------------------------------

// Two resolvers stand in for two gateway replicas: separate processes, one shared
// Redis. Deduplicating a cold prefix across them is the reason the store's create
// lock exists.
func TestResolver_CrossReplica_SingleCreate(t *testing.T) {
	expireISO := time.Now().Add(5 * time.Minute).UTC().Format(time.RFC3339)
	createResp := `{"name":"projects/p/locations/us-central1/cachedContents/new","expireTime":"` + expireISO + `","usageMetadata":{"totalTokenCount":512}}`
	fake := newFakeCacheServer(t, `{"cachedContents":[]}`, createResp, http.StatusOK)

	mr := miniredis.RunT(t)
	newReplica := func() *resolver {
		store, err := NewRedisStore(mr.Addr())
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
	store, err := NewRedisStore(mr.Addr())
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
