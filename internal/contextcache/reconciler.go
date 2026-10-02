// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package contextcache

import (
	"context"
	"fmt"
	"log/slog"
	"time"
)

// StaleThreshold is how close to its expiry an entry must be before it is treated as a
// miss, so there is time to re-resolve before the provider cache disappears underneath
// an in-flight request. The reconciler does not write entries this close to expiry.
const StaleThreshold = 10 * time.Second

// Source lists the caches a provider holds for one scope, such as a GCP project and
// region. It is the only provider-specific part of reconciliation.
type Source interface {
	// GateKey is the store key used to rate-limit rounds for this scope across the
	// fleet. It must be distinct per scope, and must not collide with cache keys.
	GateKey() string
	// List calls add once per cache the gateway created in this scope, with the cache
	// key it was created under. It should skip caches it cannot key or date. An error
	// ends the round; entries already added stay written.
	List(ctx context.Context, add func(key string, e Entry)) error
}

// Stats summarizes one reconcile round.
type Stats struct {
	// Gated is true when another replica held the gate and nothing was listed.
	Gated bool
	// Seen is the number of entries the source added.
	Seen int
	// Written is the number of entries written to the store.
	Written int
	// Skipped is the number of entries not written: already present, near expiry, or
	// failed to write.
	Skipped int
}

// Reconciler repairs drift between a Store and a provider: caches that exist at the
// provider but that the store does not know about, after a store flush or eviction, a
// dropped write, or a cache created before the store was configured.
//
// Each round:
//  1. Acquires the source's gate with SET NX and the interval as TTL, so across the fleet
//     at most one replica lists a scope per interval. A replica that loses skips the round.
//  2. Lists the source.
//  3. Writes each entry with SET NX and a TTL pinned to its expiry. SET NX means it only
//     fills gaps and never overwrites a name that a request just published.
type Reconciler struct {
	Store    Store
	Source   Source
	Interval time.Duration
	Logger   *slog.Logger
}

// Run runs rounds until ctx is done. It never returns an error: a failed round is logged
// and the next one retries.
func (rc *Reconciler) Run(ctx context.Context) {
	ticker := time.NewTicker(rc.Interval)
	defer ticker.Stop()
	for {
		rc.round(ctx)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (rc *Reconciler) round(ctx context.Context) {
	stats, err := rc.RunOnce(ctx)
	if err != nil {
		if ctx.Err() == nil {
			rc.logger().Warn("contextcache: reconcile round failed",
				slog.String("gate", rc.Source.GateKey()), slog.String("error", err.Error()))
		}
		return
	}
	if !stats.Gated {
		rc.logger().Debug("contextcache: reconcile round complete",
			slog.String("gate", rc.Source.GateKey()), slog.Int("seen", stats.Seen),
			slog.Int("written", stats.Written), slog.Int("skipped", stats.Skipped))
	}
}

// RunOnce performs one round. An error ends the round early; entries written before it
// stay written.
func (rc *Reconciler) RunOnce(ctx context.Context) (Stats, error) {
	var stats Stats
	won, err := rc.Store.AcquireGate(ctx, rc.Source.GateKey(), rc.Interval)
	if err != nil {
		return stats, fmt.Errorf("acquire reconcile gate: %w", err)
	}
	if !won {
		stats.Gated = true
		return stats, nil
	}
	err = rc.Source.List(ctx, func(key string, e Entry) {
		stats.Seen++
		if rc.write(ctx, key, e) {
			stats.Written++
		} else {
			stats.Skipped++
		}
	})
	return stats, err
}

func (rc *Reconciler) write(ctx context.Context, key string, e Entry) bool {
	// Resolvers treat entries this close to expiry as a miss, so writing one is pointless.
	ttl := time.Until(e.ExpireTime)
	if ttl < StaleThreshold {
		return false
	}
	wrote, err := rc.Store.SetNX(ctx, key, e, ttl)
	if err != nil {
		rc.logger().Warn("contextcache: reconcile store write failed",
			slog.String("name", e.Name), slog.String("error", err.Error()))
		return false
	}
	return wrote
}

func (rc *Reconciler) logger() *slog.Logger {
	if rc.Logger == nil {
		return slog.New(slog.DiscardHandler)
	}
	return rc.Logger
}
