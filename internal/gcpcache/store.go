// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package gcpcache

import (
	"context"
	"time"
)

// entry is a resolved Google cachedContents entry: the resource name and when it expires.
type entry struct {
	cacheName  string
	expireTime time.Time
}

// CacheStore is the storage layer behind the resolver, mapping a deterministic cache
// key to the Google cache name it resolved to.
//
// The store is shared across gateway replicas, which is what prevents two replicas
// from independently creating a cache for the same prefix. Implementations must be
// safe for concurrent use.
//
// Store errors are never fatal to a request: the resolver treats them as a cache miss
// and proceeds to Google. See the failure policy in the Redis cache store proposal.
type CacheStore interface {
	// Get returns the entry for key. The bool reports whether a usable entry was
	// found; a miss returns (entry{}, false, nil).
	Get(ctx context.Context, key string) (entry, bool, error)
	// Set records e under key, expiring it after ttl.
	Set(ctx context.Context, key string, e entry, ttl time.Duration) error
}

// noopStore is the CacheStore used when no shared store is configured. Every lookup
// misses and every write is discarded, so context caching becomes inert: markers are
// still parsed, but nothing is resolved or created.
//
// This is what makes "store unconfigured" behave identically to "store unreachable"
// without a nil check at each call site.
type noopStore struct{}

func (noopStore) Get(context.Context, string) (entry, bool, error) { return entry{}, false, nil }

func (noopStore) Set(context.Context, string, entry, time.Duration) error { return nil }

var _ CacheStore = noopStore{}
