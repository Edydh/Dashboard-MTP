# Dashboard MTP - Improvement Plan

## Goals

This plan focuses on the changes that will most improve the dashboard's reliability, security, and maintainability:

- Add basic engineering guardrails before large refactors
- Clarify the app's security and deployment model
- Extract a stable data layer from the monolithic app
- Modularize the dashboard incrementally without breaking behavior
- Add new features only after the foundation is stable

---

## Current State Summary

The project is a Python/Streamlit analytics dashboard for Mileage Tracker Pro, backed by Supabase (PostgreSQL). The application works, but the repository has several structural and operational risks:

- `dashboard.py` is a single 4,351-line file that mixes data access, business logic, layout, and styling
- There are no automated tests, no CI workflow, no linting, and no formatting config
- The dashboard can be exposed on the local network, but there is no real access-control layer
- `database_queries.py` exists but is not used by `dashboard.py`
- `database_queries.py` is not ready for direct reuse; parts of it are stale against the current schema and app conventions
- Documentation drifts in multiple places:
  - `README.md`, `env.example`, and `dashboard_mtp` still refer to an anon key
  - `setup.py` prompts for a service role key
  - `dashboard.py` calls `auth.admin.list_users()`, which requires elevated credentials
  - `README.md` documents fewer tabs than the app currently exposes
- `altair` is present in `requirements.txt` but is not imported anywhere in the Python code
- The project is pinned to `streamlit==1.39.0`, so newer native Streamlit patterns should not be assumed without an upgrade decision

---

## Planning Principles

- Guardrails before refactors: add linting, tests, and CI before moving large amounts of code
- Stabilize the data layer before splitting the UI: move query logic and normalization into shared modules first
- Reuse only audited code: do not wire in unused modules until they match the real schema and current app behavior
- Choose an explicit security model: local-only, internal-only, or internet-facing deployment should drive the auth solution
- Refactor incrementally: preserve a working dashboard after each phase

---

## Phase 0: Security and Deployment Model

### 0.1 Decide how the dashboard is meant to be exposed

Before adding features, decide whether the dashboard is intended to be:

- local-only for operators on the machine
- internal-only behind VPN, reverse proxy, or private network
- internet-facing for a broader audience

This decision changes what "authentication" should mean. A simple password gate in the app may be acceptable as a lightweight barrier for internal use, but it is not sufficient as the primary control for an internet-facing analytics dashboard.

### 0.2 Clarify the credential model

The current app uses `auth.admin.list_users()`, so the current runtime behavior depends on elevated Supabase credentials. The project should explicitly document:

- a service role key is currently required for the full dashboard experience
- this key must remain server-side only
- anon keys are not sufficient for the current implementation

Action items:

- update `README.md`
- update `env.example`
- update `dashboard_mtp`
- update setup prompts and validation text in `setup.py`
- add an inline warning near the auth-admin code path

### 0.3 Restrict default network exposure

If the intended use is not internet-facing, tighten the default run configuration so the app does not advertise network availability by default. This can include:

- binding Streamlit to `localhost` by default
- documenting how to opt in to LAN access
- documenting the expected reverse-proxy or VPN setup if shared access is needed

---

## Phase 1: Engineering Guardrails

### 1.1 Add linting and formatting

Add:

- `ruff` for linting and formatting
- `pyproject.toml` for tool configuration
- `.pre-commit-config.yaml` so contributors run checks locally

Benefits:

- catches obvious mistakes before runtime
- normalizes style before modularization
- reduces noisy formatting diffs during refactors

### 1.2 Add automated tests for the code paths that matter now

The first tests should target code that the running dashboard already depends on, not just generic helpers. Recommended first targets:

- data-loading and pagination behavior
- timestamp parsing and normalization
- distance and duration calculations
- revenue, growth, and retention calculations that drive visible metrics
- `supabase_client.py` behaviors with mocked responses

Suggested stack:

- `pytest`
- `pytest-mock`

Testing `utils.py` is still useful, but it should not be the only or first test target because much of it is not currently exercised by the dashboard.

### 1.3 Add CI immediately after lint and tests exist

Create `.github/workflows/ci.yml` with:

- lint job
- test job
- dependency security scan (`pip-audit` is a reasonable default)

CI should come before the large refactor, not after it. The modularization effort needs a safety net.

### 1.4 Remove unused dependencies

Remove `altair` unless a clear near-term use case exists.

Benefits:

- smaller environment
- fewer transitive dependencies
- less ambiguity about the charting stack

---

## Phase 2: Data Layer and Schema Alignment

### 2.1 Extract a shared data-access layer

Before splitting the app into many UI files, extract the shared data logic into a focused module set. The data layer should own:

- Supabase reads
- pagination
- schema normalization
- UTC/timestamp handling
- distance normalization (`actual_distance` vs `mileage`)
- duration calculation from trip fields
- common profile/auth-user joins

This is the seam that will make the later refactor safe.

### 2.2 Audit `database_queries.py` before deciding to reuse it

`database_queries.py` should not be integrated as-is. It first needs an audit because parts of it are stale relative to the actual schema and current dashboard behavior.

Observed issues to address before reuse:

- SQL text still references a `users` table in places, while the current schema uses `profiles`
- some logic assumes direct `distance` and `duration` fields instead of the app's current normalization rules
- it reads large unpaginated datasets in several methods
- it duplicates logic that is already evolving inside `dashboard.py`

Decision options:

- salvage it by rewriting the queries around the actual schema and shared data helpers
- extract only the parts that are demonstrably useful
- retire it if a cleaner shared data layer replaces it

### 2.3 Standardize analytics calculations

Move business logic that calculates dashboard metrics out of the UI layer into reusable functions with tests. This should cover:

- growth windows
- retention cohorts
- activation and conversion funnels
- subscription segmentation
- trip activity summaries

The goal is to make metrics deterministic and testable before moving them across files.

---

## Phase 3: Application Architecture

### 3.1 Modularize the app after the data layer is stable

Once the queries and calculations are extracted, split the app into a multi-file structure. A practical shape would be:

```text
dashboard/
  __init__.py
  app.py
  navigation.py
  tabs/
    overview.py
    revenue.py
    expenses.py
    growth.py
    live_feed.py
    users.py
    retention.py
    destinations.py
    advanced.py
    new_users.py
    trip_logs.py
    upgrade_signals.py
  data/
    loaders.py
    transforms.py
    metrics.py
  components/
    charts.py
    metrics.py
    styles.py
```

This should happen incrementally, tab by tab, while preserving parity with the current app.

### 3.2 Prefer modern Streamlit navigation patterns when migrating away from tabs

If the app moves beyond `st.tabs`, prefer a router-style entrypoint over a blind file split. Evaluate:

- `st.navigation` as the preferred long-term pattern
- `pages/` only if the simpler approach is sufficient

Benefits of moving off tabs:

- less initial page work
- clearer page boundaries
- shareable URLs per section
- cleaner ownership for future changes

This navigation change should be part of the architecture refactor, not treated as an isolated cosmetic feature.

### 3.3 Keep a shared shell around pages

The current app has shared elements that should remain centralized:

- page config
- sidebar filters and refresh actions
- connection status
- global styling
- shared cache invalidation behavior

Do not duplicate these concerns across page modules.

---

## Phase 4: Documentation and Operations

### 4.1 Repair documentation drift

Update repository docs so they reflect the actual runtime:

- correct the service-role requirement
- document the auth-admin dependency
- update the tab/page inventory
- align setup instructions with the real schema
- clarify local-only vs shared-access operation

### 4.2 Add a deployment artifact after the runtime is stabilized

Add a `Dockerfile` only after the entrypoint, environment variables, and run model are stable. Containerizing a moving target too early adds churn.

Once ready, the container should:

- install dependencies reproducibly
- expose the correct Streamlit port
- run the final app entrypoint
- support environment-variable based configuration

---

## Phase 5: Optional Features

These features are worthwhile, but they should come after security, tests, CI, data-layer extraction, and modularization.

### 5.1 PDF report generation

Useful for stakeholder sharing, but it should focus on the highest-value views first rather than every tab.

### 5.2 Slack or email alerts

Useful for proactive monitoring, but this likely belongs in a scheduled job or external monitor rather than inside Streamlit request handling.

### 5.3 Theme improvements

Streamlit theming is worth improving, but a custom in-app light/dark toggle should be treated as optional. First confirm whether the built-in theme behavior is already enough for the team.

---

## Recommended Implementation Order

1. Decide the dashboard exposure model and credential policy
2. Fix documentation and launcher defaults around service-role usage and network exposure
3. Add `ruff`, formatting config, and pre-commit hooks
4. Add pytest and the first targeted tests for current production code paths
5. Add GitHub Actions CI
6. Remove unused dependencies such as `altair`
7. Extract the shared data-access and metric-calculation layer
8. Audit `database_queries.py` and decide whether to salvage, rewrite, or retire it
9. Modularize the UI incrementally around the stabilized data layer
10. Migrate navigation away from `st.tabs` if the refactor still benefits from it
11. Add Docker packaging once the runtime shape is stable
12. Add optional features such as reports, alerts, and theme enhancements

---

## Task Checklist

- [ ] Decide whether the dashboard is local-only, internal-only, or internet-facing
- [ ] Restrict default network exposure if shared access is not intentional
- [ ] Document that the current dashboard requires a service role key for full functionality
- [ ] Update `README.md`, `env.example`, `setup.py`, and `dashboard_mtp` to match the real credential model
- [ ] Add a code comment near the `auth.admin.list_users()` path warning about elevated credentials
- [ ] Add `ruff`, `pyproject.toml`, and `.pre-commit-config.yaml`
- [ ] Add `pytest` and `pytest-mock`
- [ ] Write initial tests for current dashboard data loaders and metric calculations
- [ ] Write mocked tests for `supabase_client.py`
- [ ] Add GitHub Actions for linting, tests, and dependency security scanning
- [ ] Remove unused `altair` dependency unless a concrete use case is planned
- [ ] Extract shared data loading, normalization, and metric calculation modules
- [ ] Audit `database_queries.py` and decide whether to rewrite, partially reuse, or retire it
- [ ] Split the app into an entrypoint plus modular tab/page files
- [ ] Centralize shared styles, chart builders, and metric components
- [ ] Evaluate `st.navigation` versus `pages/` for the long-term navigation model
- [ ] Update repository docs to match the real app structure and operational model
- [ ] Add a `Dockerfile` after the app entrypoint and runtime contract are stable
- [ ] Revisit optional features only after the foundation is in place
