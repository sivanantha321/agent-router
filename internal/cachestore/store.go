// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

// Package cachestore provides the shared storage layer behind provider context
// caches: a mapping from a deterministic cache key to the provider cache entry it
// resolved to, plus an optional create lock that keeps replicas racing on a cold
// key from all creating at once.
//
// The package knows nothing about any particular provider. It stores an opaque
// cache name and an expiry, and arbitrates who gets to create.
package cachestore

import (
	"context"
	"time"
)

// Entry is a resolved provider cache entry: the resource name and when it expires.
type Entry struct {
	// CacheName is the provider's resource name for the cache entry.
	CacheName string
	// ExpireTime is when the provider expires the entry.
	ExpireTime time.Time
}

// Store maps a deterministic cache key to the provider cache name it resolved to.
//
// The store is shared across gateway replicas, which is what prevents two replicas
// from independently creating a cache for the same prefix. Implementations must be
// safe for concurrent use.
//
// Store errors are not expected to be fatal to a request: callers should treat them
// as a cache miss and resolve against the provider instead.
type Store interface {
	// Get returns the entry for key. The bool reports whether a usable entry was
	// found; a miss returns (Entry{}, false, nil).
	Get(ctx context.Context, key string) (Entry, bool, error)
	// Set records e under key, expiring it after ttl.
	Set(ctx context.Context, key string, e Entry, ttl time.Duration) error
}

// Locker is implemented by stores that can arbitrate cache creation across replicas.
// It is deliberately separate from Store: stores that cannot coordinate (such as
// NopStore) simply do not implement it, and callers type-assert for it.
type Locker interface {
	// TryLock claims the right to create key, reporting whether the caller won.
	// The winner must publish its result with Set, or release the claim with Unlock,
	// so that waiters are not stuck until the lock expires.
	TryLock(ctx context.Context, key string) (bool, error)
	// Unlock releases a claim that did not produce a cache name.
	Unlock(ctx context.Context, key string)
	// AwaitLeader waits for the replica holding the lock to publish its result.
	AwaitLeader(ctx context.Context, key string) (Entry, bool, error)
}

// NopStore is the Store used when no shared store is configured. Every lookup misses
// and every write is discarded, so context caching becomes inert: markers are still
// parsed, but nothing is resolved or created.
//
// This is what makes "store unconfigured" behave identically to "store unreachable"
// without a nil check at each call site. It deliberately does not implement Locker,
// so callers never coordinate through it.
type NopStore struct{}

func (NopStore) Get(context.Context, string) (Entry, bool, error) { return Entry{}, false, nil }

func (NopStore) Set(context.Context, string, Entry, time.Duration) error { return nil }

var _ Store = NopStore{}
