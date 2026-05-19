# AGENTS.md

Permanent instructions for Codex agents working in Dashboard-MTP.

## Project Role

Dashboard-MTP is a local-only Streamlit business intelligence dashboard for the Mileage Tracker Pro app. It monitors product, trip, user, retention, upgrade, expense, and RevenueCat subscription signals from Supabase.

## Operating Rules

- Keep Dashboard-MTP read-only unless the user explicitly asks for a write operation.
- Do not change MTP app behavior from this repo.
- Do not update RevenueCat entitlements, `profiles.subscription_tier`, app access, or webhook ingestion from Dashboard-MTP.
- Use the server-side Supabase service-role connection only. Never expose service-role credentials to browser/client code.
- Preserve local-only assumptions unless the user explicitly changes the deployment model.
- Prefer incremental changes with tests over broad rewrites.
- Keep business logic out of Streamlit UI when practical:
  - data loading and normalization in `dashboard_data.py`
  - analytics transforms in `dashboard_analytics.py` or `dashboard_metrics.py`
  - UI rendering in `dashboard.py`
- Use the repo virtual environment for checks:
  - `venv/bin/python -m pytest`
  - `venv/bin/python -m ruff check dashboard.py dashboard_data.py dashboard_analytics.py dashboard_metrics.py database_queries.py setup.py supabase_client.py tests`

## RevenueCat Rules

- RevenueCat webhook ingestion belongs in the MTP/Supabase backend, not in Dashboard-MTP.
- Dashboard-MTP should only read `revenuecat_events`.
- Keep production RevenueCat webhook usage in shadow-mode until production data is trusted.
- Treat `CANCELLATION` as renewal stopped, not immediate access ended.
- Treat `EXPIRATION` or `REFUND` as stronger access-ended signals.
- Before building entitlement or mismatch reports, verify production `app_user_id` values are Supabase UUIDs and not `$RCAnonymousID...`.

## Verification Expectations

For dashboard changes, run lint and tests. For visible Streamlit changes, also verify in Chrome when the dashboard is open there.
