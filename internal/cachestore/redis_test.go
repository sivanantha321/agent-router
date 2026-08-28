// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package cachestore

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

	require.NoError(t, s.Set(ctx, "k", Entry{CacheName: "projects/p/locations/r/cachedContents/x", ExpireTime: expire}, time.Minute))

	got, ok, err := s.Get(ctx, "k")
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, "projects/p/locations/r/cachedContents/x", got.CacheName)
	assert.True(t, expire.Equal(got.ExpireTime), "want %s got %s", expire, got.ExpireTime)
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

	won, err := s.TryLock(ctx, "k")
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

	first, err := s.TryLock(ctx, "k")
	require.NoError(t, err)
	assert.True(t, first)

	second, err := s.TryLock(ctx, "k")
	require.NoError(t, err)
	assert.False(t, second, "a second claim on a held key must lose")
}

// unlock must not delete a key that has moved on from the sentinel, or it would erase a
// published cache name (or another replica's fresh claim after the lock expired).
func TestRedisStore_UnlockOnlyRemovesOwnSentinel(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx := context.Background()
	expire := time.Now().Add(10 * time.Minute)

	won, err := s.TryLock(ctx, "k")
	require.NoError(t, err)
	require.True(t, won)
	require.NoError(t, s.Set(ctx, "k", Entry{CacheName: "published", ExpireTime: expire}, time.Minute))

	s.Unlock(ctx, "k")

	got, ok, err := s.Get(ctx, "k")
	require.NoError(t, err)
	require.True(t, ok, "unlock must not erase a published result")
	assert.Equal(t, "published", got.CacheName)
}

func TestRedisStore_UnlockReleasesClaim(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx := context.Background()

	won, err := s.TryLock(ctx, "k")
	require.NoError(t, err)
	require.True(t, won)
	s.Unlock(ctx, "k")

	again, err := s.TryLock(ctx, "k")
	require.NoError(t, err)
	assert.True(t, again, "a released claim must be re-claimable immediately")
}

func TestRedisStore_AwaitLeader_ReturnsPublishedResult(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx := context.Background()
	expire := time.Now().Add(10 * time.Minute).UTC().Truncate(time.Second)

	won, err := s.TryLock(ctx, "k")
	require.NoError(t, err)
	require.True(t, won)

	go func() {
		time.Sleep(100 * time.Millisecond)
		_ = s.Set(ctx, "k", Entry{CacheName: "from-leader", ExpireTime: expire}, time.Minute)
	}()

	got, ok, err := s.AwaitLeader(ctx, "k")
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, "from-leader", got.CacheName)
}

// A leader that releases without publishing must not strand its waiters for the rest of
// lockWaitTime; they are told to resolve themselves as soon as the claim disappears.
func TestRedisStore_AwaitLeader_ClaimReleasedWithoutResult(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx := context.Background()

	won, err := s.TryLock(ctx, "k")
	require.NoError(t, err)
	require.True(t, won)

	go func() {
		time.Sleep(100 * time.Millisecond)
		s.Unlock(ctx, "k")
	}()

	start := time.Now()
	_, ok, err := s.AwaitLeader(ctx, "k")
	require.ErrorIs(t, err, ErrLockHeld)
	assert.False(t, ok)
	assert.Less(t, time.Since(start), lockWaitTime, "waiter must not block for the full wait window")
}

func TestRedisStore_AwaitLeader_HonorsContextCancellation(t *testing.T) {
	s, _ := newTestRedisStore(t)
	ctx, cancel := context.WithCancel(context.Background())

	won, err := s.TryLock(ctx, "k")
	require.NoError(t, err)
	require.True(t, won)

	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	_, _, err = s.AwaitLeader(ctx, "k")
	require.ErrorIs(t, err, context.Canceled)
}

func TestEncodeDecodeEntry(t *testing.T) {
	expire := time.Now().Add(time.Hour).UTC().Truncate(time.Second)
	// The cache name is encoded last precisely because it may contain the separator.
	e := Entry{CacheName: "projects/p|weird/cachedContents/x", ExpireTime: expire}

	got, err := decodeEntry(encodeEntry(e))
	require.NoError(t, err)
	assert.Equal(t, e.CacheName, got.CacheName)
	assert.True(t, expire.Equal(got.ExpireTime))

	for _, bad := range []string{"", "no-separator", "not-a-time|name", "2020-01-01T00:00:00Z|"} {
		_, err := decodeEntry(bad)
		assert.Error(t, err, "input %q", bad)
	}
}
