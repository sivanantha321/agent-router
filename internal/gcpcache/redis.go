// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package gcpcache

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	// lockSentinel marks a key as claimed by a replica that is creating the cache.
	// A reader seeing it knows a create is in flight rather than that a name is stored.
	lockSentinel = "\x00creating"

	// lockTTL bounds how long a create lock is held. It must exceed the p99 duration of
	// a Google cachedContents create, or a second replica will claim the lock while the
	// first is still working and both will create. It is deliberately not configurable:
	// it is internal to the lock protocol, and widening the config surface later is
	// additive whereas removing a field is a breaking change.
	lockTTL = 30 * time.Second

	// lockWaitTime bounds how long a replica polls for the leader's result before giving
	// up and resolving against Google itself. It must stay well under Envoy's ext_proc
	// message timeout: a waiting request is blocked inside ProcessRequestHeaders.
	lockWaitTime = 5 * time.Second

	// lockPollInterval is how often a waiter re-reads the key while the leader works.
	lockPollInterval = 50 * time.Millisecond
)

// errLockHeld reports that another replica holds the create lock and did not publish a
// result before lockWaitTime elapsed.
var errLockHeld = errors.New("gcpcache: create lock held by another replica")

// redisStore is a CacheStore backed by Redis, shared across gateway replicas.
//
// Beyond plain get/set it arbitrates cache creation: the first replica to claim a key
// creates the Google cache while the others wait for its result, so a cold prefix
// arriving at N replicas produces one create rather than N.
type redisStore struct {
	client redis.UniversalClient
}

// NewRedisStore returns a CacheStore backed by the Redis instance at url, which accepts
// either a bare "host:port" or a full "redis://" URL.
//
// No connection is established here; go-redis dials lazily. A Redis that is unreachable
// therefore surfaces as an error from Get/Set, which the resolver treats as a miss.
func NewRedisStore(url string) (CacheStore, error) {
	opts, err := parseRedisURL(url)
	if err != nil {
		return nil, err
	}
	return &redisStore{client: redis.NewClient(opts)}, nil
}

// parseRedisURL accepts both a scheme-qualified URL and a bare host:port.
func parseRedisURL(url string) (*redis.Options, error) {
	if url == "" {
		return nil, errors.New("gcpcache: redis url is empty")
	}
	if strings.Contains(url, "://") {
		opts, err := redis.ParseURL(url)
		if err != nil {
			return nil, fmt.Errorf("gcpcache: invalid redis url: %w", err)
		}
		return opts, nil
	}
	return &redis.Options{Addr: url}, nil
}

// Get implements CacheStore. A key holding the create sentinel is reported as a miss,
// since no cache name is available yet.
func (s *redisStore) Get(ctx context.Context, key string) (entry, bool, error) {
	v, err := s.client.Get(ctx, key).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return entry{}, false, nil
		}
		return entry{}, false, fmt.Errorf("gcpcache: redis get: %w", err)
	}
	if v == lockSentinel {
		return entry{}, false, nil
	}
	e, err := decodeEntry(v)
	if err != nil {
		// A malformed value is treated as a miss rather than an error: it is not worth
		// failing a resolution over, and the next write will overwrite it.
		return entry{}, false, nil
	}
	return e, true, nil
}

// Set implements CacheStore.
func (s *redisStore) Set(ctx context.Context, key string, e entry, ttl time.Duration) error {
	if err := s.client.Set(ctx, key, encodeEntry(e), ttl).Err(); err != nil {
		return fmt.Errorf("gcpcache: redis set: %w", err)
	}
	return nil
}

// tryLock attempts to claim key for creation. It reports whether the caller won.
//
// The winner must publish its result with Set (or release the claim with unlock) so that
// waiters are not stuck until the lock expires.
func (s *redisStore) tryLock(ctx context.Context, key string) (bool, error) {
	won, err := s.client.SetNX(ctx, key, lockSentinel, lockTTL).Result()
	if err != nil {
		return false, fmt.Errorf("gcpcache: redis lock: %w", err)
	}
	return won, nil
}

// unlock releases a claim that did not produce a cache name, so waiters retry promptly
// instead of blocking for the remainder of lockTTL.
//
// It deletes only a key still holding the sentinel: if a create succeeded and published a
// real name, or the lock expired and another replica re-claimed it, the key is left alone.
func (s *redisStore) unlock(ctx context.Context, key string) {
	const script = `if redis.call("get", KEYS[1]) == ARGV[1] then return redis.call("del", KEYS[1]) end return 0`
	// Best-effort: a failure here only means waiters fall back to Google after lockWaitTime.
	_ = s.client.Eval(ctx, script, []string{key}, lockSentinel).Err()
}

// awaitLeader polls key until the leader publishes a cache name, the claim disappears, or
// lockWaitTime elapses.
//
// A timeout returns errLockHeld, which the caller treats as "resolve against Google
// yourself". That degrades a stuck leader into an extra list rather than a duplicate
// create, since the resolver lists before it creates.
func (s *redisStore) awaitLeader(ctx context.Context, key string) (entry, bool, error) {
	deadline := time.NewTimer(lockWaitTime)
	defer deadline.Stop()
	ticker := time.NewTicker(lockPollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return entry{}, false, ctx.Err()
		case <-deadline.C:
			return entry{}, false, errLockHeld
		case <-ticker.C:
			v, err := s.client.Get(ctx, key).Result()
			if err != nil {
				if errors.Is(err, redis.Nil) {
					// The leader released without publishing; caller should resolve.
					return entry{}, false, errLockHeld
				}
				return entry{}, false, fmt.Errorf("gcpcache: redis get: %w", err)
			}
			if v == lockSentinel {
				continue // Still creating.
			}
			e, decErr := decodeEntry(v)
			if decErr != nil {
				return entry{}, false, errLockHeld
			}
			return e, true, nil
		}
	}
}

// encodeEntry serializes an entry as "<RFC3339 expiry>|<cache name>". The cache name is
// last because it is the only field that may itself contain the separator.
func encodeEntry(e entry) string {
	return e.expireTime.UTC().Format(time.RFC3339) + "|" + e.cacheName
}

func decodeEntry(s string) (entry, error) {
	expiry, name, ok := strings.Cut(s, "|")
	if !ok || name == "" {
		return entry{}, fmt.Errorf("gcpcache: malformed cache entry %q", s)
	}
	t, err := time.Parse(time.RFC3339, expiry)
	if err != nil {
		return entry{}, fmt.Errorf("gcpcache: malformed cache entry expiry %q: %w", expiry, err)
	}
	return entry{cacheName: name, expireTime: t}, nil
}

var _ CacheStore = (*redisStore)(nil)
