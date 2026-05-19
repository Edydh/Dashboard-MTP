# LESSONS_LEARNED.md

Reusable technical lessons for Dashboard-MTP.

## RevenueCat Event Handling

- Normalize RevenueCat events defensively because fields may exist as direct SQL columns or nested under `raw_event.event`.
- Support both ISO timestamps and millisecond epoch values.
- Keep raw payloads inspectable in the dashboard for debugging event-shape changes.
- Always distinguish `CANCELLATION` from `EXPIRATION`:
  - `CANCELLATION` means renewal stopped.
  - `EXPIRATION` or `REFUND` is stronger evidence that access ended.

## Dashboard Architecture

- Keep external event normalization in `dashboard_data.py`.
- Keep interpretation and metric logic in `dashboard_analytics.py` or `dashboard_metrics.py`.
- Keep Streamlit UI code as thin as practical in `dashboard.py`.
- Avoid new abstractions until repeated patterns become real maintenance pressure.

## Testing and Verification

- Use `venv/bin/python`; global Python may not have project test dependencies.
- Add focused tests for every new interpretation rule.
- Browser/Chrome visual checks are important for Streamlit because tests do not catch truncated cards, missing filter options, or dense layouts.

## Supabase and Security

- Dashboard-MTP can use service-role credentials only because it is local/server-side.
- Never expose service-role credentials in browser/client code.
- For private analytics tables like `revenuecat_events`, read through server-side dashboard code only.

## Rollout Discipline

- Use shadow-mode for production RevenueCat webhook monitoring first.
- Do not sync entitlements or update profile tiers until production events and identity mapping are trusted.
- Verify production `app_user_id` values are Supabase UUIDs, not anonymous RevenueCat IDs.
