// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package gcpcache

import (
	"context"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/envoyproxy/ai-gateway/internal/filterapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
)

// The reconciler repairs drift between the store and Google: caches that exist in
// cachedContents but that the store does not know about, after a Redis flush or
// eviction, a dropped write, or a cache created before the store was configured.
//
// It never serves requests. If it is late, stuck, or stopped, requests still succeed;
// the only effect is that more of them create a cache instead of reusing one.
//
// Each round:
//  1. Acquires a per-project/region interval gate with SET NX, so across the fleet at
//     most one replica lists per interval. A replica that loses the gate skips the round.
//  2. Walks cachedContents.list page by page. The API has no server-side filter, so
//     every entry in the project and region is read.
//  3. Writes each gateway-created entry to the store with SET NX and a TTL pinned to its
//     expireTime. SET NX means it only fills gaps and never overwrites a name that a
//     request just published.

const (
	// listPageSize is the documented maximum for cachedContents.list; larger values are
	// coerced down by the API. Requesting it explicitly avoids an unspecified default.
	listPageSize = 1000

	// listMaxPages bounds one round's walk, so a misbehaving nextPageToken chain cannot
	// loop forever. Entries past the bound are picked up by the request path creating.
	listMaxPages = 10

	// reconcileGatePrefix namespaces gate keys. Cache keys are 64 hex characters, so the
	// prefix cannot collide with them.
	reconcileGatePrefix = "gcpcache:reconcile:"
)

// reconcileStore is implemented by stores that can back the reconciler. It is kept out of
// CacheStore so that the no-op store and test stores are unaffected; a store that does
// not implement it simply has no reconciler.
type reconcileStore interface {
	// setNX records e under key only if key is absent. It reports whether it wrote.
	setNX(ctx context.Context, key string, e entry, ttl time.Duration) (bool, error)
	// acquireGate claims key for ttl. It reports whether the caller won.
	acquireGate(ctx context.Context, key string, ttl time.Duration) (bool, error)
}

// cachedContentItem is the subset of a cachedContents list item the reconciler uses.
type cachedContentItem struct {
	Name        string `json:"name"`
	DisplayName string `json:"displayName"`
	ExpireTime  string `json:"expireTime"` // RFC 3339
}

type listResponse struct {
	CachedContents []cachedContentItem `json:"cachedContents"`
	NextPageToken  string              `json:"nextPageToken"`
}

// reconcileStats summarizes one round, for logging and tests.
type reconcileStats struct {
	gated   bool // Another replica held the gate; nothing was listed.
	pages   int
	seen    int
	written int
	skipped int // Entries not written: foreign, expired, or unparseable.
}

// runReconciler runs reconcile rounds for gcpAuth's project and region until ctx is
// done. It returns immediately if the store cannot back a reconciler (e.g. the no-op
// store). It never returns an error: failures are logged and the next round retries.
func (r *resolver) runReconciler(ctx context.Context, gcpAuth filterapi.GCPAuthHandler, interval time.Duration) {
	rs, ok := r.store.(reconcileStore)
	if !ok {
		return
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		r.reconcileRound(ctx, rs, gcpAuth, interval)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// reconcileRound runs one round and logs its failure, if any.
func (r *resolver) reconcileRound(ctx context.Context, rs reconcileStore, gcpAuth filterapi.GCPAuthHandler, interval time.Duration) {
	stats, err := r.reconcileOnce(ctx, rs, gcpAuth, interval)
	if err != nil {
		if ctx.Err() == nil {
			r.logger.Warn("gcpcache: reconcile round failed",
				slog.String("project", gcpAuth.GCPProject()),
				slog.String("region", gcpAuth.GCPRegion()),
				slog.String("error", err.Error()))
		}
		return
	}
	if !stats.gated {
		r.logger.Debug("gcpcache: reconcile round complete",
			slog.Int("pages", stats.pages), slog.Int("seen", stats.seen),
			slog.Int("written", stats.written), slog.Int("skipped", stats.skipped))
	}
}

// reconcileOnce performs one round. An error ends the round early; entries written
// before it stay written.
func (r *resolver) reconcileOnce(ctx context.Context, rs reconcileStore, gcpAuth filterapi.GCPAuthHandler, interval time.Duration) (reconcileStats, error) {
	var stats reconcileStats
	region, project := gcpAuth.GCPRegion(), gcpAuth.GCPProject()

	won, err := rs.acquireGate(ctx, reconcileGatePrefix+project+"/"+region, interval)
	if err != nil {
		return stats, fmt.Errorf("acquire reconcile gate: %w", err)
	}
	if !won {
		stats.gated = true
		return stats, nil
	}

	token, err := gcpAuth.GCPTokenSource().Token()
	if err != nil {
		return stats, fmt.Errorf("get GCP access token: %w", err)
	}

	base, err := url.Parse(fmt.Sprintf(gcpCachedContentsBasePath, region, project, region))
	if err != nil {
		return stats, fmt.Errorf("parse cachedContents URL: %w", err)
	}

	pageToken := ""
	for stats.pages < listMaxPages {
		lr, err := r.listPage(ctx, base, token.AccessToken, pageToken)
		if err != nil {
			return stats, err
		}
		stats.pages++
		for _, item := range lr.CachedContents {
			stats.seen++
			if r.reconcileItem(ctx, rs, item) {
				stats.written++
			} else {
				stats.skipped++
			}
		}
		// An empty token ends the list. A repeated one would loop, so treat it as the end.
		if lr.NextPageToken == "" || lr.NextPageToken == pageToken {
			return stats, nil
		}
		pageToken = lr.NextPageToken
	}
	r.logger.Warn("gcpcache: reconcile stopped at page limit; later entries were not read",
		slog.String("project", project), slog.String("region", region),
		slog.Int("maxPages", listMaxPages))
	return stats, nil
}

// reconcileItem writes one list entry to the store. It reports whether it wrote.
func (r *resolver) reconcileItem(ctx context.Context, rs reconcileStore, item cachedContentItem) bool {
	// Only entries the gateway created are keyed by a cache key. Anything else in the
	// project was created by some other client and is not ours to index.
	if !isCacheKey(item.DisplayName) || item.Name == "" {
		return false
	}
	expireTime, err := time.Parse(time.RFC3339, item.ExpireTime)
	if err != nil {
		r.logger.Warn("gcpcache: skipping cachedContents entry with unparseable expireTime",
			slog.String("name", item.Name), slog.String("expireTime", item.ExpireTime))
		return false
	}
	// storeGet treats entries this close to expiry as a miss, so writing one is pointless.
	ttl := time.Until(expireTime)
	if ttl < staleThreshold {
		return false
	}
	wrote, err := rs.setNX(ctx, item.DisplayName, entry{cacheName: item.Name, expireTime: expireTime}, ttl)
	if err != nil {
		r.logger.Warn("gcpcache: reconcile store write failed",
			slog.String("name", item.Name), slog.String("error", err.Error()))
		return false
	}
	return wrote
}

// listPage fetches one page of cachedContents. It sets the query on a copy of base, so
// the caller's URL is not altered.
func (r *resolver) listPage(ctx context.Context, base *url.URL, accessToken, pageToken string) (listResponse, error) {
	u := *base
	q := url.Values{}
	q.Set("pageSize", strconv.Itoa(listPageSize))
	if pageToken != "" {
		q.Set("pageToken", pageToken)
	}
	u.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return listResponse{}, err
	}
	req.Header.Set("Authorization", "Bearer "+accessToken)

	resp, err := r.httpClient.Do(req)
	if err != nil {
		return listResponse{}, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return listResponse{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return listResponse{}, fmt.Errorf("list cachedContents returned HTTP %d: %s", resp.StatusCode, body)
	}

	var lr listResponse
	if err = json.Unmarshal(body, &lr); err != nil {
		return listResponse{}, fmt.Errorf("failed to decode list response: %w", err)
	}
	return lr, nil
}

// isCacheKey reports whether s has the shape computeCacheKey produces: a hex-encoded
// SHA-256 digest.
func isCacheKey(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}
