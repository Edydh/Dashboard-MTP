# Dashboard Action Plan

## Purpose

This action plan turns the recently validated dashboard findings into a practical execution order.
It is narrower than `IMPROVEMENT_PLAN.md` and focuses on the highest-value reliability, UX, and maintainability fixes that should happen next.

## Guiding Principle

Fix issues that can mislead operators before investing in larger refactors.
That means prioritizing:

- incorrect or misleading metrics
- silent failure modes
- UI bugs that break trust
- performance risks in high-traffic or high-volume paths

## Phase 1: Immediate Reliability Fixes

### 1. Surface Auth Admin failures clearly

Problem:
`get_auth_users()` currently catches all exceptions and returns an empty DataFrame. This can make auth failures look like legitimate "no data" states in New Users, Retention, and any auth-enriched view.

Actions:

- Change the auth-user loading path to preserve failure state instead of silently returning an empty DataFrame.
- Show an explicit UI warning when the Auth Admin API fails or is unavailable.
- Distinguish between:
  - no auth users returned
  - auth API unavailable
  - invalid or insufficient credentials

Acceptance criteria:

- Auth failures are visible in the affected tabs.
- Operators can tell the difference between empty data and a failed auth query.
- Tabs that depend on auth data degrade gracefully instead of showing misleading zeros.

Priority:
P0

### 2. Fix misleading sidebar connection metrics

Problem:
The sidebar shows query counts, success rate, and errors from `manager.get_metrics()`, but most dashboard reads use the raw Supabase client, so those values do not represent actual dashboard traffic.

Actions:

- Either route dashboard reads through monitored manager APIs, or
- remove the misleading query counters and keep the sidebar limited to connection health and uptime.

Recommended approach:

- Short term: remove or relabel the counters if they are not trustworthy.
- Medium term: centralize reads behind monitored query helpers.

Acceptance criteria:

- Sidebar metrics match real dashboard behavior.
- Operators are not shown fake precision for query counts or success rate.

Priority:
P0

### 3. Implement real auto-refresh behavior

Problem:
The auto-refresh checkbox currently only shows an informational message. It does not trigger timed refresh behavior.

Actions:

- Implement timed refresh using a supported Streamlit approach.
- Ensure the refresh interval is explicit and stable.
- Decide whether refresh should also clear cached data or only rerun the app.

Acceptance criteria:

- Enabling auto-refresh causes the dashboard to refresh on the expected interval.
- Disabling auto-refresh stops the timed refresh behavior.
- The UI accurately reflects the implemented refresh semantics.

Priority:
P1

### 4. Fix the Advanced Analytics trip-list formatting bug

Problem:
The fuel-efficiency trip summaries currently render literal strings like `.1f`, `.2f`, and `.3f` instead of formatted values.

Actions:

- Replace the placeholder strings with actual trip-level output.
- Format each list consistently and include the most useful fields.

Suggested output:

- Most Fuel Efficient Trips: trip/user label plus MPG
- Most Expensive Trips: trip/user label plus fuel cost
- Best Value Trips: trip/user label plus cost per mile

Acceptance criteria:

- The three lists render real values.
- Formatting is consistent and readable.

Priority:
P1

## Phase 2: Product and Operational Correctness

### 5. Reduce unbounded data reads

Problem:
The dashboard still has a mix of paginated-but-unbounded reads and true full-table reads. This will degrade as trip volume grows.

Examples:

- `get_trips_dataframe()` paginates, but still reads all matching rows.
- The data-quality section performs direct `select('*')` reads on `trips` and `profiles`.

Actions:

- Add date windows where possible.
- Paginate every large-read path.
- Prefer database-side aggregates or RPCs for summary views.
- Audit the heaviest tabs first:
  - Advanced Analytics
  - Data Quality
  - user/trip summary views

Acceptance criteria:

- No large dashboard path relies on unbounded `select('*')` reads.
- High-level summary tabs avoid loading unnecessary row-level detail.

Priority:
P1

### 6. Differentiate optional-schema absence from real query failure

Problem:
Expense-table queries currently suppress exceptions. That is acceptable for optional modules, but the UI does not tell operators whether a table is intentionally absent or a real query failed.

Actions:

- Add structured handling for:
  - table not configured
  - permission/schema error
  - transient query failure
- Surface a concise status message in the Expense tab.

Acceptance criteria:

- Operators can tell whether expense data is unavailable by design or broken.
- Optional tables remain non-fatal.

Priority:
P1

### 7. Align "real-time" positioning with actual behavior

Problem:
The dashboard is described as real-time, but most cached paths use a 5-minute TTL, the live feed is 60 seconds, and timed auto-refresh is not currently implemented.

Actions:

- Audit cache TTLs by tab.
- Decide which views are truly near-real-time.
- Update UI labels and documentation to match reality.

Acceptance criteria:

- "Real-time" is used only where behavior supports it.
- Operators understand expected freshness per view.

Priority:
P2

### 8. Make fuel-price assumptions configurable

Problem:
Fuel cost analytics currently depends on a hardcoded average fuel price in the analytics layer.

Actions:

- Move the fuel-price assumption into a visible configuration control or app setting.
- Document the assumption in the UI.
- Optionally support region-specific defaults later.

Acceptance criteria:

- Operators can see and change the fuel-price assumption.
- Fuel-cost outputs are traceable to a visible input.

Priority:
P2

## Phase 3: Maintainability and Documentation

### 9. Continue extracting logic from `dashboard.py`

Problem:
`dashboard.py` still mixes UI, queries, transformation logic, and presentation in one large file.

Actions:

- Keep moving raw queries into `dashboard_data.py`.
- Keep moving derived metrics into `dashboard_metrics.py` and `dashboard_analytics.py`.
- Extract tab-specific rendering helpers one tab at a time.

Recommended next extraction targets:

- sidebar/status helpers
- Advanced Analytics rendering
- Expense tab rendering
- Upgrade Signals rendering

Acceptance criteria:

- New work avoids adding more business logic to `dashboard.py`.
- Complex tabs become easier to test and review in isolation.

Priority:
P2

### 10. Audit or retire unused modules

Problem:
`database_queries.py` and likely `utils.py` are not part of the active dashboard path and can drift into misleading or stale fallback code.

Actions:

- Confirm whether either module has a real consumer.
- If not, archive, remove, or clearly mark them as non-runtime/experimental.
- Avoid new code paths depending on them until they are audited.

Acceptance criteria:

- The active app path is clearer.
- Stale modules do not appear to be production dependencies.

Priority:
P2

### 11. Refresh stale planning and operational docs

Problem:
`IMPROVEMENT_PLAN.md` still states that the repo has no tests or CI, which is no longer true.

Actions:

- Update `IMPROVEMENT_PLAN.md` to reflect the current repo state.
- Keep documentation aligned with:
  - existing tests
  - CI workflow
  - current tab inventory
  - current auth/service-role requirements

Acceptance criteria:

- Docs describe the current repository accurately.
- Planning documents no longer contradict the codebase.

Priority:
P2

## Recommended Execution Order

1. Surface Auth Admin failures
2. Fix or remove misleading sidebar query metrics
3. Implement real auto-refresh
4. Fix the Advanced Analytics formatting bug
5. Remove unbounded full-table reads from the heaviest paths
6. Improve optional-table error visibility in Expense Analytics
7. Align freshness messaging and cache behavior
8. Make fuel-price assumptions configurable
9. Continue modularizing `dashboard.py`
10. Audit or retire unused modules
11. Refresh stale planning/docs

## Suggested Delivery Slices

### Slice A: Trust and correctness

- auth failure visibility
- sidebar metrics correction
- Advanced Analytics string bug

### Slice B: Freshness and operator UX

- real auto-refresh
- clear data-freshness messaging
- expense-source status messages

### Slice C: Scalability and maintainability

- unbounded read audit and fixes
- continued modularization
- documentation cleanup

## Definition of Done

This action plan is complete when:

- the dashboard no longer hides critical failures as empty data
- the sidebar no longer presents misleading operational metrics
- auto-refresh works as described
- the visible Advanced Analytics formatting bug is gone
- the heaviest reads are bounded or aggregated appropriately
- documentation matches the current repo and runtime behavior
