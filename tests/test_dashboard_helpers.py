import math

import pandas as pd

from dashboard_analytics import (
    build_feature_usage_summary,
    build_fuel_efficiency_analytics,
    build_revenuecat_lifecycle_summary,
    build_trip_purpose_analytics,
    build_upgrade_conversion_funnel,
    build_upgrade_purchase_summary,
)
from dashboard_data import (
    calculate_trip_duration_minutes,
    get_trip_activity_dataset,
    get_trips_dataframe,
    normalize_subscription_tier,
    prepare_global_destinations_dataset,
    prepare_revenuecat_events_dataset,
    to_week_start,
)


class DummyResponse:
    def __init__(self, data):
        self.data = data


class DummyQuery:
    def __init__(self, rows):
        self.rows = rows
        self.start = 0
        self.end = len(rows)
        self.created_at_gte = None

    def select(self, _columns):
        return self

    def order(self, _column, desc=False):
        assert desc is False
        return self

    def gte(self, column, value):
        assert column == "created_at"
        self.created_at_gte = value
        return self

    def range(self, start, end):
        self.start = start
        self.end = end
        return self

    def execute(self):
        rows = self.rows
        if self.created_at_gte is not None:
            rows = [row for row in rows if row.get("created_at") >= self.created_at_gte]
        return DummyResponse(rows[self.start:self.end + 1])


class DummySupabase:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return DummyQuery(self.tables[name])


def test_normalize_subscription_tier_maps_empty_values_to_free():
    series = pd.Series([None, "", " Premium ", "BASIC", "null", "None"])

    normalized = normalize_subscription_tier(series)

    assert normalized.tolist() == ["free", "free", "premium", "basic", "free", "free"]


def test_to_week_start_returns_monday_midnight_utc():
    values = pd.Series(["2026-04-09T15:30:00Z", "2026-04-06T01:00:00Z"])

    week_starts = to_week_start(values)

    assert week_starts.dt.strftime("%Y-%m-%d %H:%M:%S%z").tolist() == [
        "2026-04-06 00:00:00+0000",
        "2026-04-06 00:00:00+0000",
    ]


def test_calculate_trip_duration_minutes_handles_invalid_and_non_positive_values():
    start_values = pd.Series(
        ["2026-04-09T10:00:00Z", "2026-04-09T11:00:00Z", "not-a-timestamp"]
    )
    end_values = pd.Series(
        ["2026-04-09T10:45:00Z", "2026-04-09T11:00:00Z", "2026-04-09T12:00:00Z"]
    )

    durations = calculate_trip_duration_minutes(start_values, end_values)

    assert durations.iloc[0] == 45
    assert math.isnan(durations.iloc[1])
    assert math.isnan(durations.iloc[2])


def test_prepare_global_destinations_dataset_parses_mixed_iso_timestamps():
    destinations_df = pd.DataFrame(
        [
            {
                "description": "Toronto, Canada",
                "latitude": "43.6532",
                "longitude": "-79.3832",
                "usage_count": "7",
                "created_at": "2026-04-22T17:19:14+00:00",
                "updated_at": "2026-04-22T17:19:14.123456+00:00",
                "last_used_at": "2026-04-23T09:05:01+00:00",
            },
            {
                "description": "Invalid Coordinates",
                "latitude": None,
                "longitude": "-79.0000",
                "usage_count": "2",
                "created_at": "2026-04-22T17:19:14.999999+00:00",
                "updated_at": "2026-04-22T17:19:15+00:00",
                "last_used_at": "2026-04-23T09:05:02.654321+00:00",
            },
        ]
    )

    prepared_df = prepare_global_destinations_dataset(destinations_df)

    assert prepared_df["description"].tolist() == ["Toronto, Canada"]
    assert prepared_df["usage_count"].tolist() == [7]
    assert prepared_df.iloc[0]["created_at"].isoformat() == "2026-04-22T17:19:14+00:00"
    assert prepared_df.iloc[0]["updated_at"].isoformat() == "2026-04-22T17:19:14.123456+00:00"
    assert prepared_df.iloc[0]["last_used_at"].isoformat() == "2026-04-23T09:05:01+00:00"


def test_prepare_revenuecat_events_dataset_normalizes_raw_payload_fields():
    event_timestamp_ms = int(pd.Timestamp("2026-04-10T12:00:00Z").timestamp() * 1000)
    purchased_at_ms = int(pd.Timestamp("2026-04-10T11:58:00Z").timestamp() * 1000)
    expiration_at_ms = int(pd.Timestamp("2026-05-10T11:58:00Z").timestamp() * 1000)
    events_df = pd.DataFrame(
        [
            {
                "id": "row-1",
                "created_at": "2026-04-10T12:01:00Z",
                "raw_event": {
                    "event": {
                        "id": "evt-1",
                        "type": "INITIAL_PURCHASE",
                        "app_user_id": "user-1",
                        "aliases": ["anon-1", "user-1"],
                        "product_id": "mtp_pro_monthly",
                        "entitlement_ids": ["pro"],
                        "store": "APP_STORE",
                        "environment": "SANDBOX",
                        "transaction_id": "tx-1",
                        "original_transaction_id": "otx-1",
                        "purchased_at_ms": purchased_at_ms,
                        "expiration_at_ms": expiration_at_ms,
                        "event_timestamp_ms": event_timestamp_ms,
                    }
                },
            }
        ]
    )

    prepared_df = prepare_revenuecat_events_dataset(events_df)
    event = prepared_df.iloc[0]

    assert event["revenuecat_event_id"] == "evt-1"
    assert event["event_type"] == "INITIAL_PURCHASE"
    assert event["app_user_id"] == "user-1"
    assert event["alias_display"] == "anon-1, user-1"
    assert event["product_id"] == "mtp_pro_monthly"
    assert event["entitlement_display"] == "pro"
    assert event["store"] == "app_store"
    assert event["environment"] == "sandbox"
    assert event["transaction_id"] == "tx-1"
    assert event["original_transaction_id"] == "otx-1"
    assert event["event_timestamp"].isoformat() == "2026-04-10T12:00:00+00:00"
    assert event["purchased_at"].isoformat() == "2026-04-10T11:58:00+00:00"
    assert event["expiration_at"].isoformat() == "2026-05-10T11:58:00+00:00"


def test_prepare_revenuecat_events_dataset_prefers_direct_columns_and_sorts_latest_first():
    events_df = pd.DataFrame(
        [
            {
                "id": "row-older",
                "revenuecat_event_id": "evt-older",
                "event_type": "initial_purchase",
                "app_user_id": "user-older",
                "store": "play_store",
                "environment": "production",
                "event_timestamp": "2026-04-09T12:00:00Z",
                "raw_event": {},
            },
            {
                "id": "row-newer",
                "revenuecat_event_id": "evt-newer",
                "event_type": "renewal",
                "app_user_id": "user-newer",
                "store": "app_store",
                "environment": "production",
                "event_timestamp": "2026-04-10T12:00:00Z",
                "raw_event": {},
            },
        ]
    )

    prepared_df = prepare_revenuecat_events_dataset(events_df)

    assert prepared_df["revenuecat_event_id"].tolist() == ["evt-newer", "evt-older"]
    assert prepared_df["event_type"].tolist() == ["RENEWAL", "INITIAL_PURCHASE"]


def test_build_revenuecat_lifecycle_summary_marks_expired_after_cancel_and_expiration():
    events_df = pd.DataFrame(
        [
            {
                "revenuecat_event_id": "evt-1",
                "event_type": "INITIAL_PURCHASE",
                "app_user_id": "user-1",
                "product_id": "monthly",
                "entitlement_display": "pro",
                "store": "play_store",
                "environment": "sandbox",
                "transaction_id": "tx-1",
                "event_timestamp": "2026-05-19T14:22:47Z",
            },
            {
                "revenuecat_event_id": "evt-2",
                "event_type": "CANCELLATION",
                "app_user_id": "user-1",
                "product_id": "monthly",
                "entitlement_display": "pro",
                "store": "play_store",
                "environment": "sandbox",
                "transaction_id": "tx-1",
                "event_timestamp": "2026-05-19T14:30:18Z",
            },
            {
                "revenuecat_event_id": "evt-3",
                "event_type": "EXPIRATION",
                "app_user_id": "user-1",
                "product_id": "monthly",
                "entitlement_display": "pro",
                "store": "play_store",
                "environment": "sandbox",
                "transaction_id": "tx-1",
                "event_timestamp": "2026-05-19T14:30:19Z",
            },
        ]
    )

    timeline_df, summary = build_revenuecat_lifecycle_summary(events_df, app_user_id="user-1")

    assert timeline_df["stage"].tolist() == ["Purchase", "Cancellation", "Expiration"]
    assert summary["events"] == 3
    assert summary["purchase_events"] == 1
    assert summary["cancellation_events"] == 1
    assert summary["expiration_events"] == 1
    assert summary["current_state"] == "Expired"
    assert summary["latest_event_type"] == "EXPIRATION"


def test_build_revenuecat_lifecycle_summary_keeps_cancellation_distinct_from_expiration():
    events_df = pd.DataFrame(
        [
            {
                "revenuecat_event_id": "evt-1",
                "event_type": "INITIAL_PURCHASE",
                "app_user_id": "user-1",
                "event_timestamp": "2026-05-19T14:22:47Z",
            },
            {
                "revenuecat_event_id": "evt-2",
                "event_type": "CANCELLATION",
                "app_user_id": "user-1",
                "event_timestamp": "2026-05-19T14:30:18Z",
            },
        ]
    )

    _, summary = build_revenuecat_lifecycle_summary(events_df, app_user_id="user-1")

    assert summary["current_state"] == "Cancelled"
    assert "access may remain active until expiration" in summary["state_detail"]
    assert summary["expiration_events"] == 0


def test_build_upgrade_conversion_funnel_counts_only_paywall_users_downstream():
    events_df = pd.DataFrame(
        [
            {"id": "1", "user_id": "u1", "event_name": "paywall_opened", "feature_display": "Exports"},
            {"id": "2", "user_id": "u1", "event_name": "purchase_started", "feature_display": "Exports"},
            {"id": "3", "user_id": "u1", "event_name": "purchase_completed", "feature_display": "Exports"},
            {"id": "4", "user_id": "u2", "event_name": "paywall_viewed", "feature_display": "Routes"},
            {"id": "5", "user_id": "u3", "event_name": "purchase_started", "feature_display": "Routes"},
            {"id": "6", "user_id": "u4", "event_name": "paywall_displayed", "feature_display": "Exports"},
        ]
    )

    funnel_df, paywall_feature_df, has_completed_purchase_events = build_upgrade_conversion_funnel(
        events_df
    )

    assert funnel_df["Stage"].tolist() == [
        "Paywall Opened",
        "Purchase Started",
        "Purchase Completed",
    ]
    assert funnel_df["Users"].tolist() == [3, 1, 1]
    assert funnel_df["Events"].tolist() == [3, 1, 1]
    assert has_completed_purchase_events is True

    assert paywall_feature_df.to_dict("records") == [
        {"Feature": "Exports", "Paywall Opens": 2, "Users": 2},
        {"Feature": "Routes", "Paywall Opens": 1, "Users": 1},
    ]


def test_build_upgrade_purchase_summary_counts_generic_and_inferred_completions():
    events_df = pd.DataFrame(
        [
            {"id": "1", "user_id": "u1", "event_name": "subscription_purchased"},
            {"id": "2", "user_id": "u2", "event_name": "premium_purchase_started", "subscription_tier": "pro_lifetime"},
            {"id": "3", "user_id": "u3", "event_name": "checkout_started", "subscription_tier": "free"},
            {"id": "4", "user_id": "u4", "event_name": "paywall_opened", "subscription_tier": "pro_lifetime"},
        ]
    )

    purchase_summary = build_upgrade_purchase_summary(events_df)

    assert purchase_summary == {
        "purchase_start_users": 2,
        "purchase_start_events": 2,
        "purchase_complete_event_users": 1,
        "purchase_complete_events": 1,
        "inferred_purchase_complete_users": 1,
        "observed_purchase_complete_users": 2,
    }


def test_build_feature_usage_summary_splits_free_and_premium_usage():
    events_df = pd.DataFrame(
        [
            {
                "id": "1",
                "user_id": "u1",
                "event_name": "trip_exported",
                "feature_display": "Exports",
                "subscription_tier": "premium",
                "event_ts": "2026-04-09T10:00:00Z",
            },
            {
                "id": "2",
                "user_id": "u1",
                "event_name": "paywall_opened",
                "feature_display": "Exports",
                "subscription_tier": "premium",
                "event_ts": "2026-04-09T11:00:00Z",
            },
            {
                "id": "3",
                "user_id": "u2",
                "event_name": "trip_exported",
                "feature_display": "Exports",
                "subscription_tier": "free",
                "event_ts": "2026-04-09T12:00:00Z",
            },
            {
                "id": "4",
                "user_id": "u3",
                "event_name": "purchase_started",
                "feature_display": "Insights",
                "subscription_tier": "basic",
                "event_ts": "2026-04-08T09:00:00Z",
            },
        ]
    )

    summary_df = build_feature_usage_summary(events_df)

    exports_row = summary_df.loc[summary_df["Feature"] == "Exports"].iloc[0]
    insights_row = summary_df.loc[summary_df["Feature"] == "Insights"].iloc[0]

    assert exports_row["Events"] == 3
    assert exports_row["Unique Users"] == 2
    assert exports_row["Free Users"] == 1
    assert exports_row["Premium Users"] == 1
    assert exports_row["Free Events"] == 1
    assert exports_row["Premium Events"] == 2
    assert exports_row["Paywall Opens"] == 1
    assert exports_row["Purchase Starts"] == 0
    assert exports_row["Last Seen"].isoformat() == "2026-04-09T12:00:00+00:00"

    assert insights_row["Free Users"] == 0
    assert insights_row["Premium Users"] == 1
    assert insights_row["Purchase Starts"] == 1


def test_build_trip_purpose_analytics_classifies_common_purpose_patterns():
    trips_df = pd.DataFrame(
        [
            {
                "purpose": "Client meeting downtown",
                "user_id": "u1",
                "distance": 12,
                "fuel_used": 1.5,
                "reimbursement": 10,
                "created_at": "2026-04-09T10:00:00Z",
            },
            {
                "purpose": "Family shopping run",
                "user_id": "u2",
                "distance": 8,
                "fuel_used": 1.0,
                "reimbursement": 0,
                "created_at": "2026-04-09T12:00:00Z",
            },
            {
                "purpose": None,
                "user_id": "u3",
                "distance": 5,
                "fuel_used": 0.5,
                "reimbursement": 0,
                "created_at": "2026-04-09T14:00:00Z",
            },
        ]
    )

    purpose_df = build_trip_purpose_analytics(trips_df)

    assert purpose_df["purpose"].tolist() == [
        "Client meeting downtown",
        "Family shopping run",
        "Not Specified",
    ]
    assert purpose_df["purpose_category"].tolist() == [
        "Business",
        "Personal",
        "Other",
    ]


def test_build_fuel_efficiency_analytics_calculates_trip_level_costs():
    trips_df = pd.DataFrame(
        [
            {
                "distance": 100,
                "fuel_used": 4,
                "reimbursement": 12,
                "user_id": "u1",
                "created_at": "2026-04-09T10:00:00Z",
                "purpose": "Business",
            },
            {
                "distance": 0,
                "fuel_used": 2,
                "reimbursement": 0,
                "user_id": "u2",
                "created_at": "2026-04-09T11:00:00Z",
                "purpose": "Personal",
            },
        ]
    )

    fuel_df = build_fuel_efficiency_analytics(trips_df, avg_fuel_price=4.00)

    assert fuel_df["mpg"].tolist() == [25.0, 0.0]
    assert fuel_df["fuel_cost"].tolist() == [16.0, 8.0]
    assert fuel_df["cost_per_mile"].tolist() == [0.16, 0.0]


def test_get_trips_dataframe_paginates_across_multiple_batches():
    trip_rows = [
        {"id": str(index), "user_id": f"u{index}", "created_at": f"2026-04-01T00:{index % 60:02d}:00Z"}
        for index in range(1001)
    ]
    supabase = DummySupabase({"trips": trip_rows})

    trips_df = get_trips_dataframe(supabase, "id, user_id, created_at")

    assert len(trips_df) == 1001
    assert trips_df.iloc[0]["id"] == "0"
    assert trips_df.iloc[-1]["id"] == "1000"


def test_get_trip_activity_dataset_normalizes_distance_and_completion_status():
    trip_rows = [
        {
            "id": "1",
            "user_id": "u1",
            "created_at": "2026-04-01T00:00:00Z",
            "status": "Completed",
            "mileage": 10,
            "actual_distance": 12.5,
        },
        {
            "id": "2",
            "user_id": "u2",
            "created_at": "2026-04-01T01:00:00Z",
            "status": "complated",
            "mileage": 9,
            "actual_distance": None,
        },
        {
            "id": "3",
            "user_id": "u3",
            "created_at": None,
            "status": "planned",
            "mileage": 4,
            "actual_distance": 5,
        },
    ]
    supabase = DummySupabase({"trips": trip_rows})

    trips_df = get_trip_activity_dataset(supabase)

    assert trips_df["id"].tolist() == ["1", "2"]
    assert trips_df["distance"].tolist() == [12.5, 9.0]
    assert trips_df["status_norm"].tolist() == ["completed", "complated"]
    assert trips_df["is_completed"].tolist() == [True, True]
