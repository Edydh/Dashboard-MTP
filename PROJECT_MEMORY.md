# PROJECT_MEMORY.md

Long-term context for Dashboard-MTP.

## Project Summary

Dashboard-MTP is a Python/Streamlit dashboard backed by Supabase. It monitors the Mileage Tracker Pro app, including users, trips, activity, retention, growth, expenses, upgrade intent, and RevenueCat subscription signals.

The dashboard is currently local-only and uses server-side Supabase credentials. Full functionality requires a service-role key because the dashboard reads Supabase Auth Admin users and private analytics tables.

## Current Architecture

- `dashboard.py`: main Streamlit app and tab rendering.
- `dashboard_data.py`: shared data-access and normalization helpers.
- `dashboard_metrics.py`: product metrics, retention, growth, revenue, risk summaries.
- `dashboard_analytics.py`: upgrade analytics, feature usage, trip purpose/fuel analytics, RevenueCat lifecycle interpretation.
- `supabase_client.py`: Supabase client/manager, health checks, retry/rate-limit support.
- `tests/`: pytest coverage for metrics, helpers, and Supabase manager behavior.

## RevenueCat Integration State

Dashboard-MTP has a read-only `💳 RevenueCat` tab backed by the `revenuecat_events` Supabase table.

Current RevenueCat tab capabilities:

- webhook activity health
- latest event age
- events in last 24 hours
- sandbox/production event counts
- event counts by type, store, and environment
- latest 50 events
- raw payload inspection
- selected-user purchase lifecycle

The tested sandbox user is:

```text
1a1a8187-6916-4248-865c-c42271f6393d
```

That user has verified sandbox lifecycle events:

- `INITIAL_PURCHASE`
- `CANCELLATION`
- `EXPIRATION`

The lifecycle view correctly shows this user as `Expired`.

## Important Constraints

- Dashboard-MTP is not the RevenueCat ingestion service.
- Do not update subscription state from Dashboard-MTP.
- Do not change MTP app gating from Dashboard-MTP.
- Keep RevenueCat production webhook in shadow-mode until production events are inspected and trusted.

## Current Branch Context

Recent RevenueCat work was done on:

```text
codex/fix-destinations-map-timestamps
```

Recent pushed commits:

- `33a297c Add RevenueCat webhook monitor`
- `a5bde8b Add RevenueCat purchase lifecycle view`
- `6bd0723 Always show production RevenueCat filter`
