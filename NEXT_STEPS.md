# NEXT_STEPS.md

Current execution queue for Dashboard-MTP.

## Immediate

1. Monitor the `💳 RevenueCat` tab after production webhook delivery is enabled.
2. Confirm `Sandbox/Prod` changes from `5 / 0` to production count greater than `0`.
3. Inspect the first production raw payload.
4. Confirm production `app_user_id` values are Supabase UUIDs and not `$RCAnonymousID...`.
5. Confirm production `store`, `product_id`, `entitlement_ids`, and `environment` normalize correctly.

## Next Dashboard Improvements

1. Add profile join for RevenueCat events:
   - app user ID
   - profile name
   - email when available
   - current `profiles.subscription_tier`
2. Add production-only lifecycle view/filter once production events exist.
3. Add mismatch reports:
   - RevenueCat active signal but Supabase profile tier is free
   - Supabase profile premium but RevenueCat expired/refunded
4. Add paywall-to-RevenueCat funnel:
   - paywall opened
   - purchase started
   - RevenueCat purchase observed
   - cancellation/expiration/refund follow-up

## Later

1. Consider a derived `revenuecat_subscription_status` table or view after production shadow-mode is stable.
2. Add cohort-level monetization metrics:
   - signup cohort
   - activation cohort
   - trip behavior before purchase
   - feature usage before purchase
3. Continue extracting large UI sections from `dashboard.py` only when there is a clear testability or maintenance win.

## Do Not Do Yet

- Do not update app access behavior.
- Do not update `profiles.subscription_tier`.
- Do not sync entitlements from Dashboard-MTP.
- Do not treat cancellation as immediate access loss without expiration/refund evidence.
