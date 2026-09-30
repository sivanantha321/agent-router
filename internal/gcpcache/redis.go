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

// redisStore is a CacheStore backed by Redis, shared across gateway replicas.
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

// Get implements CacheStore.
func (s *redisStore) Get(ctx context.Context, key string) (entry, bool, error) {
	v, err := s.client.Get(ctx, key).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return entry{}, false, nil
		}
		return entry{}, false, fmt.Errorf("gcpcache: redis get: %w", err)
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

// setNX implements reconcileStore.
func (s *redisStore) setNX(ctx context.Context, key string, e entry, ttl time.Duration) (bool, error) {
	wrote, err := s.client.SetNX(ctx, key, encodeEntry(e), ttl).Result()
	if err != nil {
		return false, fmt.Errorf("gcpcache: redis setnx: %w", err)
	}
	return wrote, nil
}

// acquireGate implements reconcileStore.
func (s *redisStore) acquireGate(ctx context.Context, key string, ttl time.Duration) (bool, error) {
	won, err := s.client.SetNX(ctx, key, "1", ttl).Result()
	if err != nil {
		return false, fmt.Errorf("gcpcache: redis gate: %w", err)
	}
	return won, nil
}

var _ reconcileStore = (*redisStore)(nil)

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
