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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var reconcileAuth = &fakeGCPAuth{token: "tok", region: "us-central1", project: "p"}

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

	stats, err := r.reconcileOnce(context.Background(), store, reconcileAuth, time.Minute)
	require.NoError(t, err)
	assert.Equal(t, 2, stats.pages)
	assert.Equal(t, 2, stats.written)
	assert.Equal(t, int64(2), calls.Load())
	for _, q := range *queries {
		assert.Contains(t, q, "pageSize=1000")
	}
	for n, name := range map[int]string{1: "c/1", 2: "c/2"} {
		e, ok, err := store.Get(context.Background(), testKey(n))
		require.NoError(t, err)
		require.True(t, ok, "key %d", n)
		assert.Equal(t, name, e.cacheName)
	}
}

func TestReconcile_DoesNotOverwriteExistingEntry(t *testing.T) {
	exp := time.Now().Add(5 * time.Minute)
	srv, _, _ := newListServer(t, map[string]string{
		"": `{"cachedContents":[` + itemJSON(testKey(1), "c/from-list", exp) + `]}`,
	})
	store, _ := newTestRedisStore(t)
	require.NoError(t, store.Set(context.Background(), testKey(1), entry{cacheName: "c/published", expireTime: exp}, time.Minute))
	r := resolverWithServer(srv.URL, store)

	stats, err := r.reconcileOnce(context.Background(), store, reconcileAuth, time.Minute)
	require.NoError(t, err)
	assert.Equal(t, 0, stats.written)
	e, _, _ := store.Get(context.Background(), testKey(1))
	assert.Equal(t, "c/published", e.cacheName, "reconciler must not clobber a request's write")
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

	stats, err := r.reconcileOnce(context.Background(), store, reconcileAuth, time.Minute)
	require.NoError(t, err)
	assert.Equal(t, listMaxPages, stats.pages)
	assert.Equal(t, int64(listMaxPages), calls.Load())
}

func TestReconcile_RepeatedTokenEndsWalk(t *testing.T) {
	srv, calls, _ := newListServer(t, map[string]string{
		"":     `{"cachedContents":[],"nextPageToken":"same"}`,
		"same": `{"cachedContents":[],"nextPageToken":"same"}`,
	})
	store, _ := newTestRedisStore(t)
	r := resolverWithServer(srv.URL, store)

	_, err := r.reconcileOnce(context.Background(), store, reconcileAuth, time.Minute)
	require.NoError(t, err)
	assert.Equal(t, int64(2), calls.Load())
}

func TestReconcile_GateSkipsSecondReplica(t *testing.T) {
	srv, calls, _ := newListServer(t, map[string]string{"": `{"cachedContents":[]}`})
	store, mr := newTestRedisStore(t)
	second, err := NewRedisStore(mr.Addr())
	require.NoError(t, err)
	a := resolverWithServer(srv.URL, store)
	b := resolverWithServer(srv.URL, second)

	statsA, err := a.reconcileOnce(context.Background(), store, reconcileAuth, time.Minute)
	require.NoError(t, err)
	assert.False(t, statsA.gated)
	statsB, err := b.reconcileOnce(context.Background(), second.(*redisStore), reconcileAuth, time.Minute)
	require.NoError(t, err)
	assert.True(t, statsB.gated)
	assert.Equal(t, int64(1), calls.Load())

	// Once the interval passes, a replica may reconcile again.
	mr.FastForward(time.Minute + time.Second)
	statsB, err = b.reconcileOnce(context.Background(), second.(*redisStore), reconcileAuth, time.Minute)
	require.NoError(t, err)
	assert.False(t, statsB.gated)
}

func TestReconcile_GateIsPerProjectRegion(t *testing.T) {
	srv, calls, _ := newListServer(t, map[string]string{"": `{"cachedContents":[]}`})
	store, _ := newTestRedisStore(t)
	r := resolverWithServer(srv.URL, store)
	other := &fakeGCPAuth{token: "tok", region: "europe-west1", project: "p"}

	_, err := r.reconcileOnce(context.Background(), store, reconcileAuth, time.Minute)
	require.NoError(t, err)
	stats, err := r.reconcileOnce(context.Background(), store, other, time.Minute)
	require.NoError(t, err)
	assert.False(t, stats.gated, "one region's round must not suppress another's")
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

	stats, err := r.reconcileOnce(context.Background(), store, reconcileAuth, time.Minute)
	require.NoError(t, err)
	assert.Equal(t, 4, stats.seen)
	assert.Equal(t, 1, stats.written)
	assert.Equal(t, 3, stats.skipped)
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

	_, err := r.reconcileOnce(context.Background(), store, reconcileAuth, time.Minute)
	require.NoError(t, err)
	assert.InDelta(t, (5 * time.Minute).Seconds(), mr.TTL(testKey(1)).Seconds(), 5)
}

func TestReconcile_ListFailureReturnsErrorWithoutPanicking(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)
	store, _ := newTestRedisStore(t)
	r := resolverWithServer(srv.URL, store)

	_, err := r.reconcileOnce(context.Background(), store, reconcileAuth, time.Minute)
	require.ErrorContains(t, err, "HTTP 500")
	// reconcileRound is the caller the loop uses; it must swallow the error.
	r.reconcileRound(context.Background(), store, reconcileAuth, time.Minute)
}

func TestRunReconciler_NoopStoreReturnsImmediately(t *testing.T) {
	r := New(nil, nil, nil).(*resolver)
	done := make(chan struct{})
	go func() {
		r.runReconciler(context.Background(), reconcileAuth, time.Minute)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("runReconciler must return at once for a store that cannot reconcile")
	}
}

func TestRunReconciler_StopsOnContextCancel(t *testing.T) {
	srv, _, _ := newListServer(t, map[string]string{"": `{"cachedContents":[]}`})
	store, _ := newTestRedisStore(t)
	r := resolverWithServer(srv.URL, store)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		r.runReconciler(ctx, reconcileAuth, time.Hour)
		close(done)
	}()
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("runReconciler must return when its context is cancelled")
	}
}

func TestIsCacheKey(t *testing.T) {
	assert.True(t, isCacheKey(testKey(7)))
	assert.False(t, isCacheKey(""))
	assert.False(t, isCacheKey(strings.Repeat("z", 64)))
	assert.False(t, isCacheKey(testKey(7)[:63]))
}
