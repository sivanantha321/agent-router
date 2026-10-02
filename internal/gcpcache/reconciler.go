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

	"github.com/envoyproxy/ai-gateway/internal/contextcache"
	"github.com/envoyproxy/ai-gateway/internal/filterapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
)

// This file supplies the GCP side of reconciliation: a contextcache.Source that walks
// cachedContents.list for one project and region. The loop, gate, and store writes live
// in contextcache.Reconciler.

const (
	// listPageSize is the documented maximum for cachedContents.list; larger values are
	// coerced down by the API. Requesting it explicitly avoids an unspecified default.
	listPageSize = 1000

	// listMaxPages bounds one round's walk, so a misbehaving nextPageToken chain cannot
	// loop forever. Entries past the bound are picked up by the request path creating.
	listMaxPages = 10

	// reconcileGatePrefix namespaces gate keys. Cache keys are 64 hex characters, so the
	// prefix cannot collide with them. It is unchanged from before the store moved to
	// contextcache, so old and new replicas share gates during a rolling deploy.
	reconcileGatePrefix = "gcpcache:reconcile:"
)

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

// newReconciler returns a reconciler for gcpAuth's project and region.
func (r *resolver) newReconciler(gcpAuth filterapi.GCPAuthHandler, interval time.Duration) *contextcache.Reconciler {
	return &contextcache.Reconciler{
		Store:    r.store,
		Source:   &listSource{r: r, auth: gcpAuth},
		Interval: interval,
		Logger:   r.logger,
	}
}

// listSource is a contextcache.Source over cachedContents.list.
type listSource struct {
	r    *resolver
	auth filterapi.GCPAuthHandler
	// pages is the number of pages read by the last List call, for tests.
	pages int
}

// GateKey implements contextcache.Source.
func (s *listSource) GateKey() string {
	return reconcileGatePrefix + s.auth.GCPProject() + "/" + s.auth.GCPRegion()
}

// List implements contextcache.Source. The API has no server-side filter, so the walk
// reads every entry in the project and region; only gateway-created ones are added.
func (s *listSource) List(ctx context.Context, add func(string, contextcache.Entry)) error {
	s.pages = 0
	region, project := s.auth.GCPRegion(), s.auth.GCPProject()

	token, err := s.auth.GCPTokenSource().Token()
	if err != nil {
		return fmt.Errorf("get GCP access token: %w", err)
	}
	base, err := url.Parse(fmt.Sprintf(gcpCachedContentsBasePath, region, project, region))
	if err != nil {
		return fmt.Errorf("parse cachedContents URL: %w", err)
	}

	pageToken := ""
	for s.pages < listMaxPages {
		lr, err := s.r.listPage(ctx, base, token.AccessToken, pageToken)
		if err != nil {
			return err
		}
		s.pages++
		for _, item := range lr.CachedContents {
			if key, e, ok := s.toEntry(item); ok {
				add(key, e)
			}
		}
		// An empty token ends the list. A repeated one would loop, so treat it as the end.
		if lr.NextPageToken == "" || lr.NextPageToken == pageToken {
			return nil
		}
		pageToken = lr.NextPageToken
	}
	s.r.logger.Warn("gcpcache: reconcile stopped at page limit; later entries were not read",
		slog.String("project", project), slog.String("region", region),
		slog.Int("maxPages", listMaxPages))
	return nil
}

// toEntry converts a list item. It reports false for items the gateway did not create
// (their displayName is not a cache key) and for items whose expiry does not parse.
func (s *listSource) toEntry(item cachedContentItem) (string, contextcache.Entry, bool) {
	if !isCacheKey(item.DisplayName) || item.Name == "" {
		return "", contextcache.Entry{}, false
	}
	expireTime, err := time.Parse(time.RFC3339, item.ExpireTime)
	if err != nil {
		s.r.logger.Warn("gcpcache: skipping cachedContents entry with unparseable expireTime",
			slog.String("name", item.Name), slog.String("expireTime", item.ExpireTime))
		return "", contextcache.Entry{}, false
	}
	return item.DisplayName, contextcache.Entry{Name: item.Name, ExpireTime: expireTime}, true
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
