// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package gcpcache

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/envoyproxy/ai-gateway/internal/contextcache"
)

var reconcileAuth = &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}

// newTestRedisStore starts an in-process Redis and returns a store pointed at it.
func newTestRedisStore(t *testing.T) (contextcache.ReconcileStore, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.RunT(t)
	s, err := contextcache.NewRedisStore(mr.Addr())
	require.NoError(t, err)
	return s, mr
}

// reconcileOnce runs one round for auth against r's store and returns its stats and the
// number of list pages read.
func reconcileOnce(t *testing.T, r *resolver, auth *fakeGCPAuth) (contextcache.Stats, int, error) {
	t.Helper()
	rc, ok := r.newReconciler(auth, time.Minute)
	require.True(t, ok)
	stats, err := rc.RunOnce(context.Background())
	return stats, rc.Source.(*listSource).pages, err
}

func testKey(n int) string { return fmt.Sprintf("%064x", n) }

func itemJSON(key, name string, expire time.Time) string {
	return fmt.Sprintf(`{"name":%q,"displayName":%q,"expireTime":%q}`, name, key, expire.UTC().Format(time.RFC3339))
}

// newListServer serves pages keyed by pageToken ("" is the first page). It fails the test
// on any non-GET, and records every query it saw.
func newListServer(t *testing.T, pages map[string]string) (*httptest.Server, *atomic.Int64, *[]string) {
	t.Helper()
	var calls atomic.Int64
	var queries []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("reconciler issued %s; it must only list", r.Method)
		}
		calls.Add(1)
		queries = append(queries, r.URL.RawQuery)
		body, ok := pages[r.URL.Query().Get("pageToken")]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &calls, &queries
}

func TestReconcile_MultiPageWalkWritesEveryEntry(t *testing.T) {
	exp := time.Now().Add(5 * time.Minute)
	srv, calls, queries := newListServer(t, map[string]string{
		"":   `{"cachedContents":[` + itemJSON(testKey(1), "c/1", exp) + `],"nextPageToken":"p2"}`,
		"p2": `{"cachedContents":[` + itemJSON(testKey(2), "c/2", exp) + `]}`,
	})
	store, _ := newTestRedisStore(t)
	r := resolverWithServer(srv.URL, store)

	stats, pages, err := reconcileOnce(t, r, reconcileAuth)
	require.NoError(t, err)
	assert.Equal(t, 2, pages)
	assert.Equal(t, 2, stats.Written)
	assert.Equal(t, int64(2), calls.Load())
	for _, q := range *queries {
		assert.Contains(t, q, "pageSize=1000")
	}
	for n, name := range map[int]string{1: "c/1", 2: "c/2"} {
		e, ok, err := store.Get(context.Background(), testKey(n))
		require.NoError(t, err)
		require.True(t, ok, "key %d", n)
		assert.Equal(t, name, e.Name)
	}
}

func TestReconcile_DoesNotOverwriteExistingEntry(t *testing.T) {
	exp := time.Now().Add(5 * time.Minute)
	srv, _, _ := newListServer(t, map[string]string{
		"": `{"cachedContents":[` + itemJSON(testKey(1), "c/from-list", exp) + `]}`,
	})
	store, _ := newTestRedisStore(t)
	require.NoError(t, store.Set(context.Background(), testKey(1), contextcache.Entry{Name: "c/published", ExpireTime: exp}, time.Minute))
	r := resolverWithServer(srv.URL, store)

	stats, _, err := reconcileOnce(t, r, reconcileAuth)
	require.NoError(t, err)
	assert.Equal(t, 0, stats.Written)
	e, _, _ := store.Get(context.Background(), testKey(1))
	assert.Equal(t, "c/published", e.Name, "reconciler must not clobber a request's write")
}

func TestReconcile_PageLimitStopsEndlessChain(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n := calls.Add(1)
		_, _ = fmt.Fprintf(w, `{"cachedContents":[],"nextPageToken":"t%d"}`, n)
	}))
	t.Cleanup(srv.Close)
	store, _ := newTestRedisStore(t)
	r := resolverWithServer(srv.URL, store)

	_, pages, err := reconcileOnce(t, r, reconcileAuth)
	require.NoError(t, err)
	assert.Equal(t, listMaxPages, pages)
	assert.Equal(t, int64(listMaxPages), calls.Load())
}

func TestReconcile_RepeatedTokenEndsWalk(t *testing.T) {
	srv, calls, _ := newListServer(t, map[string]string{
		"":     `{"cachedContents":[],"nextPageToken":"same"}`,
		"same": `{"cachedContents":[],"nextPageToken":"same"}`,
	})
	store, _ := newTestRedisStore(t)
	r := resolverWithServer(srv.URL, store)

	_, _, err := reconcileOnce(t, r, reconcileAuth)
	require.NoError(t, err)
	assert.Equal(t, int64(2), calls.Load())
}

func TestReconcile_GateSkipsSecondReplica(t *testing.T) {
	srv, calls, _ := newListServer(t, map[string]string{"": `{"cachedContents":[]}`})
	store, mr := newTestRedisStore(t)
	second, err := contextcache.NewRedisStore(mr.Addr())
	require.NoError(t, err)
	a := resolverWithServer(srv.URL, store)
	b := resolverWithServer(srv.URL, second)

	statsA, _, err := reconcileOnce(t, a, reconcileAuth)
	require.NoError(t, err)
	assert.False(t, statsA.Gated)
	statsB, _, err := reconcileOnce(t, b, reconcileAuth)
	require.NoError(t, err)
	assert.True(t, statsB.Gated)
	assert.Equal(t, int64(1), calls.Load())

	// Once the interval passes, a replica may reconcile again.
	mr.FastForward(time.Minute + time.Second)
	statsB, _, err = reconcileOnce(t, b, reconcileAuth)
	require.NoError(t, err)
	assert.False(t, statsB.Gated)
}

func TestReconcile_GateIsPerProjectRegion(t *testing.T) {
	srv, calls, _ := newListServer(t, map[string]string{"": `{"cachedContents":[]}`})
	store, _ := newTestRedisStore(t)
	r := resolverWithServer(srv.URL, store)
	other := &fakeGCPAuth{token: "tok", region: "europe-west1", project: "p"}

	_, _, err := reconcileOnce(t, r, reconcileAuth)
	require.NoError(t, err)
	stats, _, err := reconcileOnce(t, r, other)
	require.NoError(t, err)
	assert.False(t, stats.Gated, "one region's round must not suppress another's")
	assert.Equal(t, int64(2), calls.Load())
}

func TestReconcile_SkipsForeignExpiredAndUnparseable(t *testing.T) {
	exp := time.Now().Add(5 * time.Minute)
	items := []string{
		itemJSON("my-own-cache", "c/foreign", exp),                         // not a cache key
		itemJSON(testKey(1), "c/near-expiry", time.Now().Add(time.Second)), // within staleThreshold
		fmt.Sprintf(`{"name":"c/bad","displayName":%q,"expireTime":"not-a-time"}`, testKey(2)),
		itemJSON(testKey(3), "c/good", exp),
	}
	srv, _, _ := newListServer(t, map[string]string{"": `{"cachedContents":[` + strings.Join(items, ",") + `]}`})
	store, mr := newTestRedisStore(t)
	r := resolverWithServer(srv.URL, store)

	stats, _, err := reconcileOnce(t, r, reconcileAuth)
	require.NoError(t, err)
	// Foreign and unparseable items are dropped by the source; the near-expiry one is
	// seen but not written.
	assert.Equal(t, 2, stats.Seen)
	assert.Equal(t, 1, stats.Written)
	assert.Equal(t, 1, stats.Skipped)
	assert.False(t, mr.Exists("my-own-cache"))
	assert.False(t, mr.Exists(testKey(1)))
	assert.False(t, mr.Exists(testKey(2)))
	assert.True(t, mr.Exists(testKey(3)))
}

func TestReconcile_EntryTTLMatchesExpiry(t *testing.T) {
	exp := time.Now().Add(5 * time.Minute)
	srv, _, _ := newListServer(t, map[string]string{"": `{"cachedContents":[` + itemJSON(testKey(1), "c/1", exp) + `]}`})
	store, mr := newTestRedisStore(t)
	r := resolverWithServer(srv.URL, store)

	_, _, err := reconcileOnce(t, r, reconcileAuth)
	require.NoError(t, err)
	assert.InDelta(t, (5 * time.Minute).Seconds(), mr.TTL(testKey(1)).Seconds(), 5)
}

func TestReconcile_ListFailureReturnsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)
	store, _ := newTestRedisStore(t)
	r := resolverWithServer(srv.URL, store)

	_, _, err := reconcileOnce(t, r, reconcileAuth)
	require.ErrorContains(t, err, "HTTP 500")
}

func TestNewReconciler_NoopStoreHasNone(t *testing.T) {
	r := New(nil, nil, nil).(*resolver)
	_, ok := r.newReconciler(reconcileAuth, time.Minute)
	assert.False(t, ok, "a store that cannot reconcile must not get a reconciler")
}

func TestIsCacheKey(t *testing.T) {
	assert.True(t, isCacheKey(testKey(7)))
	assert.False(t, isCacheKey(""))
	assert.False(t, isCacheKey(strings.Repeat("z", 64)))
	assert.False(t, isCacheKey(testKey(7)[:63]))
}
