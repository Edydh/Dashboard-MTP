import pandas as pd

from dashboard_metrics import (
    build_growth_metrics,
    build_revenue_metrics,
    build_subscription_segment_behavior_summary,
    build_trip_statistics,
    build_usage_patterns,
    build_user_retention_outputs,
    build_users_at_risk_summary,
)


def test_build_user_retention_outputs_calculates_activation_and_weekly_retention():
    profiles_df = pd.DataFrame(
        [
            {
                "id": "u1",
                "full_name": "Free User",
                "phone_number": "111",
                "subscription_tier": "free",
                "subscription_tier_norm": "free",
                "subscription_segment": "Free",
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "id": "u2",
                "full_name": "Premium User",
                "phone_number": "222",
                "subscription_tier": "premium",
                "subscription_tier_norm": "premium",
                "subscription_segment": "Premium",
                "created_at": "2026-01-01T00:00:00Z",
            },
        ]
    )
    auth_users_df = pd.DataFrame(
        [
            {
                "id": "u1",
                "email": "u1@example.com",
                "created_at": "2026-01-01T00:00:00Z",
                "email_confirmed_at": "2026-01-01T00:05:00Z",
                "last_sign_in_at": "2026-01-15T00:00:00Z",
            }
        ]
    )
    trips_df = pd.DataFrame(
        [
            {"id": "t1", "user_id": "u1", "created_at": "2026-01-02T00:00:00Z", "is_completed": True},
            {"id": "t2", "user_id": "u1", "created_at": "2026-01-09T00:00:00Z", "is_completed": True},
            {"id": "t3", "user_id": "u2", "created_at": "2026-01-10T00:00:00Z", "is_completed": False},
            {"id": "t4", "user_id": "u2", "created_at": "2026-01-11T00:00:00Z", "is_completed": True},
            {"id": "t5", "user_id": "u2", "created_at": "2026-01-12T00:00:00Z", "is_completed": True},
        ]
    )

    summary, activation_breakdown_df, cohort_summary_df, weekly_retention_df, activation_dist_df = (
        build_user_retention_outputs(
            profiles_df,
            auth_users_df,
            trips_df,
            now_utc=pd.Timestamp("2026-02-15T00:00:00Z"),
        )
    )

    assert summary == {
        "total_registered": 2,
        "activation_1_trip_users": 2,
        "activation_1_trip_rate": 100.0,
        "activation_3_trip_users": 1,
        "activation_3_trip_rate": 50.0,
        "activation_7_day_users": 1,
        "activation_7_day_rate": 50.0,
        "week_1_retention": 100.0,
        "week_4_retention": 0.0,
        "week_1_eligible_users": 2,
        "week_4_eligible_users": 2,
        "median_days_to_first_trip": 5.0,
    }
    assert activation_breakdown_df["users"].tolist() == [2, 1, 1]
    assert cohort_summary_df["total_users"].tolist() == [2]
    assert cohort_summary_df["rate_1_trip"].tolist() == [100.0]
    assert cohort_summary_df["rate_3_trips"].tolist() == [50.0]
    assert cohort_summary_df["rate_7_day"].tolist() == [50.0]

    week_one_df = weekly_retention_df.loc[weekly_retention_df["weeks_since_registration"] == 1]
    assert week_one_df["active_users"].tolist() == [2]
    assert week_one_df["retention_rate"].tolist() == [100.0]
    assert activation_dist_df["days_to_first_trip"].tolist() == [1, 9]


def test_build_subscription_segment_behavior_summary_filters_trips_to_lookback_window():
    profiles_df = pd.DataFrame(
        [
            {"id": "u1", "subscription_segment": "Free"},
            {"id": "u2", "subscription_segment": "Free"},
            {"id": "u3", "subscription_segment": "Premium"},
        ]
    )
    trips_df = pd.DataFrame(
        [
            {
                "id": "t1",
                "user_id": "u1",
                "created_at": "2026-04-05T00:00:00Z",
                "is_completed": True,
                "distance": 10,
            },
            {
                "id": "t2",
                "user_id": "u2",
                "created_at": "2026-02-01T00:00:00Z",
                "is_completed": True,
                "distance": 5,
            },
            {
                "id": "t3",
                "user_id": "u3",
                "created_at": "2026-04-01T00:00:00Z",
                "is_completed": True,
                "distance": 20,
            },
            {
                "id": "t4",
                "user_id": "u3",
                "created_at": "2026-04-02T00:00:00Z",
                "is_completed": True,
                "distance": 25,
            },
            {
                "id": "t5",
                "user_id": "u3",
                "created_at": "2026-04-03T00:00:00Z",
                "is_completed": False,
                "distance": 30,
            },
        ]
    )
    events_df = pd.DataFrame(
        [
            {"id": "e1", "user_id": "u1", "event_name": "paywall_opened"},
            {"id": "e2", "user_id": "u3", "event_name": "purchase_started"},
            {"id": "e3", "user_id": "u3", "event_name": "purchase_started"},
            {"id": "e4", "user_id": "u3", "event_name": "trip_exported"},
        ]
    )

    summary_df = build_subscription_segment_behavior_summary(
        profiles_df,
        trips_df,
        events_df,
        lookback_days=30,
        now_utc=pd.Timestamp("2026-04-10T00:00:00Z"),
    )

    free_row = summary_df.loc[summary_df["Segment"] == "Free"].iloc[0]
    premium_row = summary_df.loc[summary_df["Segment"] == "Premium"].iloc[0]

    assert free_row["Users"] == 2
    assert free_row["Active Users"] == 1
    assert free_row["Active Rate"] == 50.0
    assert free_row["Avg Trips/User"] == 0.5
    assert free_row["Avg Completed Trips/User"] == 0.5
    assert free_row["3+ Trip Users"] == 0
    assert free_row["Tracked Events"] == 1
    assert free_row["Paywall Open Users"] == 1
    assert free_row["Purchase Start Users"] == 0

    assert premium_row["Users"] == 1
    assert premium_row["Active Users"] == 1
    assert premium_row["Active Rate"] == 100.0
    assert premium_row["Avg Trips/User"] == 3.0
    assert premium_row["Avg Completed Trips/User"] == 2.0
    assert premium_row["3+ Trip Users"] == 1
    assert premium_row["3+ Trip Rate"] == 100.0
    assert premium_row["Tracked Events"] == 3
    assert premium_row["Avg Tracked Events/User"] == 3.0
    assert premium_row["Paywall Open Users"] == 0
    assert premium_row["Purchase Start Users"] == 1


def test_build_usage_patterns_extracts_daily_and_hourly_dimensions():
    trips_df = pd.DataFrame(
        [
            {"id": "t1", "user_id": "u1", "created_at": "2026-04-10T13:45:00Z"},
            {"id": "t2", "user_id": "u2", "created_at": "2026-04-11T00:15:00Z"},
        ]
    )

    usage_df = build_usage_patterns(trips_df)

    assert usage_df["date"].astype(str).tolist() == ["2026-04-10", "2026-04-11"]
    assert usage_df["hour"].tolist() == [13, 0]
    assert usage_df["day_of_week"].tolist() == ["Friday", "Saturday"]


def test_build_trip_statistics_calculates_recent_trip_windows():
    trips_df = pd.DataFrame(
        [
            {
                "id": "t1",
                "created_at": "2026-04-10T08:00:00Z",
                "distance": 10,
                "duration": 20,
            },
            {
                "id": "t2",
                "created_at": "2026-04-05T08:00:00Z",
                "distance": 30,
                "duration": 40,
            },
            {
                "id": "t3",
                "created_at": "2026-03-15T08:00:00Z",
                "distance": 5,
                "duration": 10,
            },
        ]
    )

    stats = build_trip_statistics(
        trips_df,
        now_utc=pd.Timestamp("2026-04-10T12:00:00Z"),
    )

    assert stats == {
        "total_trips": 3,
        "total_distance": 45.0,
        "avg_distance": 15.0,
        "total_duration": 70.0,
        "avg_duration": 70.0 / 3,
        "trips_today": 1,
        "trips_this_week": 2,
        "trips_this_month": 3,
    }


def test_build_users_at_risk_summary_returns_old_trip_users_and_never_used_count():
    profiles_df = pd.DataFrame(
        [
            {"id": "u1", "full_name": "Old User", "phone_number": "111", "subscription_tier": "free"},
            {"id": "u2", "full_name": "Recent User", "phone_number": "222", "subscription_tier": "premium"},
            {"id": "u3", "full_name": "Never Used", "phone_number": "333", "subscription_tier": "basic"},
        ]
    )
    trips_df = pd.DataFrame(
        [
            {"id": "t1", "user_id": "u1", "created_at": "2026-03-01T00:00:00Z", "distance": 12},
            {"id": "t2", "user_id": "u2", "created_at": "2026-04-08T00:00:00Z", "distance": 8},
        ]
    )

    at_risk_df, never_used_count = build_users_at_risk_summary(
        profiles_df,
        trips_df,
        inactivity_days=14,
        now_utc=pd.Timestamp("2026-04-10T00:00:00Z"),
    )

    assert never_used_count == 1
    assert at_risk_df["full_name"].tolist() == ["Old User"]
    assert at_risk_df["trip_count"].tolist() == [1]
    assert at_risk_df["total_distance"].tolist() == [12.0]
    assert at_risk_df["days_since_last_trip"].tolist() == [40]


def test_build_revenue_metrics_normalizes_tier_names_before_pricing():
    profiles_df = pd.DataFrame(
        [
            {"subscription_tier": "Premium"},
            {"subscription_tier": " basic "},
            {"subscription_tier": None},
        ]
    )

    metrics = build_revenue_metrics(profiles_df)

    assert metrics["total_users"] == 3
    assert metrics["mrr"] == 29.98
    assert metrics["arr"] == 359.76
    assert round(metrics["arpu"], 2) == 9.99
    assert metrics["tier_counts"] == {"premium": 1, "basic": 1, "free": 1}
    assert metrics["tier_distribution"]["premium"]["count"] == 1
    assert metrics["tier_distribution"]["basic"]["revenue"] == 9.99


def test_build_growth_metrics_calculates_period_windows_and_daily_signups():
    profiles_df = pd.DataFrame(
        [
            {"created_at": "2026-04-09T00:00:00Z"},
            {"created_at": "2026-04-08T00:00:00Z"},
            {"created_at": "2026-04-02T00:00:00Z"},
            {"created_at": "2026-03-20T00:00:00Z"},
            {"created_at": "2026-01-15T00:00:00Z"},
        ]
    )
    trips_df = pd.DataFrame(
        [
            {"created_at": "2026-04-09T00:00:00Z"},
            {"created_at": "2026-04-04T00:00:00Z"},
            {"created_at": "2026-04-01T00:00:00Z"},
        ]
    )

    metrics = build_growth_metrics(
        profiles_df,
        trips_df,
        now_utc=pd.Timestamp("2026-04-10T00:00:00Z"),
    )

    assert metrics["wow"]["users"] == {"current": 2, "previous": 1, "growth_rate": 100.0}
    assert metrics["wow"]["trips"] == {"current": 2, "previous": 1, "growth_rate": 100.0}
    assert metrics["mom"]["users"]["current"] == 4
    assert metrics["mom"]["users"]["previous"] == 0
    assert metrics["daily_signups"]["signups"].tolist() == [1, 1, 1, 1, 1]
    assert metrics["daily_signups"]["cumulative"].tolist() == [1, 2, 3, 4, 5]
