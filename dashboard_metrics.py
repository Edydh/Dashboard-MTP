"""
Shared dashboard metric builders that operate on prepared DataFrames.
"""

from typing import Optional

import numpy as np
import pandas as pd

from dashboard_data import normalize_subscription_tier, to_week_start


def _empty_retention_outputs():
    empty_activation = pd.DataFrame(columns=["milestone", "users", "rate"])
    empty_cohort_summary = pd.DataFrame(
        columns=[
            "registration_week",
            "total_users",
            "users_1_trip",
            "users_3_trips",
            "users_7_day",
            "rate_1_trip",
            "rate_3_trips",
            "rate_7_day",
        ]
    )
    empty_weekly = pd.DataFrame()
    empty_activation_dist = pd.DataFrame(columns=["days_to_first_trip", "users"])
    return {}, empty_activation, empty_cohort_summary, empty_weekly, empty_activation_dist


def _ensure_columns(df: pd.DataFrame, columns) -> pd.DataFrame:
    result = df.copy() if not df.empty else pd.DataFrame()
    for col in columns:
        if col not in result.columns:
            result[col] = None
    return result


def build_usage_patterns(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare trip timestamps for usage heatmaps and daily activity charts."""
    usage_df = _ensure_columns(trips_df, ["created_at", "user_id"])
    if usage_df.empty:
        return pd.DataFrame()

    usage_df["created_at"] = pd.to_datetime(usage_df["created_at"], errors="coerce", utc=True)
    usage_df = usage_df.dropna(subset=["created_at"]).copy()
    if usage_df.empty:
        return pd.DataFrame()

    usage_df["date"] = usage_df["created_at"].dt.date
    usage_df["hour"] = usage_df["created_at"].dt.hour
    usage_df["day_of_week"] = usage_df["created_at"].dt.day_name()
    return usage_df


def build_trip_statistics(trips_df: pd.DataFrame, now_utc: Optional[pd.Timestamp] = None):
    """Calculate overall trip totals, averages, and recent-window counts."""
    stats_df = _ensure_columns(trips_df, ["id", "created_at", "distance", "duration"])
    if stats_df.empty:
        return {}

    stats_df["created_at"] = pd.to_datetime(stats_df["created_at"], errors="coerce", utc=True)
    stats_df["distance"] = pd.to_numeric(stats_df["distance"], errors="coerce").fillna(0)
    stats_df["duration"] = pd.to_numeric(stats_df["duration"], errors="coerce")
    stats_df = stats_df.dropna(subset=["created_at"]).copy()
    if stats_df.empty:
        return {}

    now_utc = now_utc or pd.Timestamp.now(tz="UTC")
    created_at_utc = stats_df["created_at"]

    return {
        "total_trips": int(len(stats_df)),
        "total_distance": float(stats_df["distance"].sum()),
        "avg_distance": float(stats_df["distance"].mean()),
        "total_duration": float(stats_df["duration"].sum()),
        "avg_duration": float(stats_df["duration"].mean()),
        "trips_today": int((created_at_utc.dt.date == now_utc.date()).sum()),
        "trips_this_week": int((created_at_utc >= now_utc - pd.Timedelta(days=7)).sum()),
        "trips_this_month": int((created_at_utc >= now_utc - pd.Timedelta(days=30)).sum()),
    }


def build_users_at_risk_summary(
    profiles_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    inactivity_days: int = 14,
    now_utc: Optional[pd.Timestamp] = None,
):
    """Return users whose last trip falls outside the inactivity threshold."""
    empty_at_risk_df = pd.DataFrame(
        columns=[
            "full_name",
            "phone_number",
            "subscription_tier",
            "trip_count",
            "total_distance",
            "last_trip",
            "days_since_last_trip",
        ]
    )
    profile_base = _ensure_columns(
        profiles_df,
        ["id", "full_name", "phone_number", "subscription_tier"],
    )
    if profile_base.empty:
        return empty_at_risk_df, 0

    profile_base["id"] = profile_base["id"].astype(str)
    trip_base = _ensure_columns(trips_df, ["user_id", "created_at", "distance"])
    if trip_base.empty:
        agg = pd.DataFrame(columns=["user_id", "trip_count", "last_trip", "total_distance"])
    else:
        trip_base["user_id"] = trip_base["user_id"].astype(str)
        trip_base["created_at"] = pd.to_datetime(trip_base["created_at"], errors="coerce", utc=True)
        trip_base["distance"] = pd.to_numeric(trip_base["distance"], errors="coerce").fillna(0)
        trip_base = trip_base.dropna(subset=["created_at"]).copy()
        agg = (
            trip_base.groupby("user_id")
            .agg(
                trip_count=("user_id", "count"),
                last_trip=("created_at", "max"),
                total_distance=("distance", "sum"),
            )
            .reset_index()
        )

    merged = profile_base.merge(agg, left_on="id", right_on="user_id", how="left")
    now_utc = now_utc or pd.Timestamp.now(tz="UTC")
    merged["last_trip"] = pd.to_datetime(merged["last_trip"], errors="coerce", utc=True)
    merged["days_since_last_trip"] = (now_utc - merged["last_trip"]).dt.days

    never_used_count = int(merged["last_trip"].isna().sum())
    at_risk_df = merged[
        merged["days_since_last_trip"].notna() & (merged["days_since_last_trip"] > inactivity_days)
    ].copy()
    if at_risk_df.empty:
        return empty_at_risk_df, never_used_count

    at_risk_df = at_risk_df[
        [
            "full_name",
            "phone_number",
            "subscription_tier",
            "trip_count",
            "total_distance",
            "last_trip",
            "days_since_last_trip",
        ]
    ].copy()
    at_risk_df["total_distance"] = at_risk_df["total_distance"].fillna(0).round(1)
    at_risk_df["trip_count"] = (
        pd.to_numeric(at_risk_df["trip_count"], errors="coerce").fillna(0).astype(int)
    )
    at_risk_df = at_risk_df.sort_values(
        ["days_since_last_trip", "trip_count"], ascending=[False, False]
    ).reset_index(drop=True)
    return at_risk_df, never_used_count


def build_revenue_metrics(profiles_df: pd.DataFrame, pricing=None):
    """Calculate recurring revenue metrics from subscription tiers."""
    tier_base = _ensure_columns(
        profiles_df,
        ["subscription_tier", "subscription_tier_norm"],
    )
    if tier_base.empty:
        return {}

    pricing = pricing or {
        "free": 0,
        "basic": 9.99,
        "premium": 19.99,
        "pro": 39.99,
        "enterprise": 99.99,
    }

    if tier_base["subscription_tier_norm"].notna().any():
        tiers = tier_base["subscription_tier_norm"]
    else:
        tiers = normalize_subscription_tier(tier_base["subscription_tier"])
    tier_counts = tiers.fillna("free").astype(str).str.strip().str.lower().value_counts().to_dict()

    total_users = int(len(tier_base))
    mrr = sum(tier_counts.get(tier, 0) * price for tier, price in pricing.items())
    arr = mrr * 12
    arpu = mrr / total_users if total_users > 0 else 0

    tier_distribution = {
        tier: {
            "count": int(tier_counts.get(tier, 0)),
            "revenue": round(float(tier_counts.get(tier, 0) * price), 2),
            "percentage": (tier_counts.get(tier, 0) / total_users * 100) if total_users > 0 else 0,
        }
        for tier, price in pricing.items()
    }

    return {
        "mrr": round(float(mrr), 2),
        "arr": round(float(arr), 2),
        "arpu": round(float(arpu), 2),
        "total_users": total_users,
        "tier_distribution": tier_distribution,
        "tier_counts": tier_counts,
    }


def _calculate_period_growth(df: pd.DataFrame, days_current: int, days_previous: int, now_utc):
    if df.empty:
        return {"current": 0, "previous": 0, "growth_rate": 0}

    period_df = _ensure_columns(df, ["created_at"])
    period_df["created_at"] = pd.to_datetime(period_df["created_at"], errors="coerce", utc=True)
    period_df = period_df.dropna(subset=["created_at"]).copy()
    if period_df.empty:
        return {"current": 0, "previous": 0, "growth_rate": 0}

    current_start = now_utc - pd.Timedelta(days=days_current)
    previous_start = now_utc - pd.Timedelta(days=days_previous)
    previous_end = current_start

    current_count = int((period_df["created_at"] >= current_start).sum())
    previous_count = int(
        (
            (period_df["created_at"] >= previous_start)
            & (period_df["created_at"] < previous_end)
        ).sum()
    )
    if previous_count > 0:
        growth_rate = ((current_count - previous_count) / previous_count) * 100
    else:
        growth_rate = 100 if current_count > 0 else 0

    return {
        "current": current_count,
        "previous": previous_count,
        "growth_rate": growth_rate,
    }


def build_growth_metrics(
    profiles_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    now_utc: Optional[pd.Timestamp] = None,
):
    """Calculate user and trip growth windows plus daily signup trend data."""
    users_df = _ensure_columns(profiles_df, ["created_at"])[["created_at"]].copy()
    users_df["created_at"] = pd.to_datetime(users_df["created_at"], errors="coerce", utc=True)
    users_df = users_df.dropna(subset=["created_at"]).copy()
    if users_df.empty:
        return {}

    trip_frame = _ensure_columns(trips_df, ["created_at"])[["created_at"]].copy()
    trip_frame["created_at"] = pd.to_datetime(trip_frame["created_at"], errors="coerce", utc=True)
    trip_frame = trip_frame.dropna(subset=["created_at"]).copy()

    now_utc = now_utc or pd.Timestamp.now(tz="UTC")
    wow_users = _calculate_period_growth(users_df, 7, 14, now_utc)
    mom_users = _calculate_period_growth(users_df, 30, 60, now_utc)
    qoq_users = _calculate_period_growth(users_df, 90, 180, now_utc)

    if trip_frame.empty:
        zero_growth = {"current": 0, "previous": 0, "growth_rate": 0}
        wow_trips = zero_growth
        mom_trips = zero_growth
        qoq_trips = zero_growth
    else:
        wow_trips = _calculate_period_growth(trip_frame, 7, 14, now_utc)
        mom_trips = _calculate_period_growth(trip_frame, 30, 60, now_utc)
        qoq_trips = _calculate_period_growth(trip_frame, 90, 180, now_utc)

    daily_signups = users_df.assign(date=users_df["created_at"].dt.date).groupby("date").size().reset_index(name="signups")
    daily_signups["cumulative"] = daily_signups["signups"].cumsum()

    return {
        "wow": {"users": wow_users, "trips": wow_trips},
        "mom": {"users": mom_users, "trips": mom_trips},
        "qoq": {"users": qoq_users, "trips": qoq_trips},
        "daily_signups": daily_signups.tail(30).reset_index(drop=True),
    }


def build_user_retention_outputs(
    profiles_df: pd.DataFrame,
    auth_users_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    now_utc: Optional[pd.Timestamp] = None,
):
    """Build activation and weekly retention outputs from prepared user/trip datasets."""
    empty_outputs = _empty_retention_outputs()
    now_utc = now_utc or pd.Timestamp.now(tz="UTC")

    profile_columns = [
        "id",
        "full_name",
        "phone_number",
        "subscription_tier",
        "subscription_tier_norm",
        "subscription_segment",
        "created_at",
    ]
    auth_columns = ["id", "email", "created_at", "email_confirmed_at", "last_sign_in_at"]
    trip_columns = ["id", "user_id", "created_at", "is_completed"]

    profile_base = _ensure_columns(profiles_df, profile_columns)[profile_columns].copy()
    profile_base.rename(columns={"created_at": "profile_created_at"}, inplace=True)

    auth_base = _ensure_columns(auth_users_df, auth_columns)[auth_columns].copy()
    auth_base.rename(columns={"created_at": "auth_created_at"}, inplace=True)

    if profile_base.empty and auth_base.empty:
        return empty_outputs

    if profile_base.empty:
        users_df = auth_base.copy()
        users_df["full_name"] = None
        users_df["phone_number"] = None
        users_df["subscription_tier"] = "free"
        users_df["subscription_tier_norm"] = "free"
        users_df["subscription_segment"] = "Free"
        users_df["profile_created_at"] = pd.NaT
    elif auth_base.empty:
        users_df = profile_base.copy()
        users_df["email"] = None
        users_df["auth_created_at"] = pd.NaT
        users_df["email_confirmed_at"] = pd.NaT
        users_df["last_sign_in_at"] = pd.NaT
    else:
        users_df = profile_base.merge(auth_base, on="id", how="outer")

    users_df["profile_created_at"] = pd.to_datetime(
        users_df["profile_created_at"], errors="coerce", utc=True
    )
    users_df["auth_created_at"] = pd.to_datetime(
        users_df["auth_created_at"], errors="coerce", utc=True
    )
    users_df["registration_at"] = users_df["auth_created_at"].combine_first(users_df["profile_created_at"])
    users_df = users_df.dropna(subset=["registration_at"]).copy()

    if users_df.empty:
        return empty_outputs

    users_df["subscription_tier"] = users_df["subscription_tier"].fillna("free")
    users_df["subscription_tier_norm"] = normalize_subscription_tier(users_df["subscription_tier"])
    users_df["subscription_segment"] = np.where(
        users_df["subscription_tier_norm"].eq("free"),
        "Free",
        "Premium",
    )
    users_df["registration_week"] = to_week_start(users_df["registration_at"])

    trips_df = _ensure_columns(trips_df, trip_columns)
    if trips_df.empty:
        total_registered = int(len(users_df))
        summary = {
            "total_registered": total_registered,
            "activation_1_trip_users": 0,
            "activation_1_trip_rate": 0.0,
            "activation_3_trip_users": 0,
            "activation_3_trip_rate": 0.0,
            "activation_7_day_users": 0,
            "activation_7_day_rate": 0.0,
            "week_1_retention": 0.0,
            "week_4_retention": 0.0,
            "week_1_eligible_users": 0,
            "week_4_eligible_users": 0,
            "median_days_to_first_trip": None,
        }
        activation_breakdown_df = pd.DataFrame(
            [
                {"milestone": "1+ Trips", "users": 0, "rate": 0.0},
                {"milestone": "3+ Trips", "users": 0, "rate": 0.0},
                {"milestone": "1st Trip in 7 Days", "users": 0, "rate": 0.0},
            ]
        )
        return summary, activation_breakdown_df, empty_outputs[2], empty_outputs[3], empty_outputs[4]

    ordered_trips = trips_df.copy()
    ordered_trips["created_at"] = pd.to_datetime(ordered_trips["created_at"], errors="coerce", utc=True)
    ordered_trips = ordered_trips.dropna(subset=["created_at"]).sort_values(["user_id", "created_at"]).copy()
    if ordered_trips.empty:
        return build_user_retention_outputs(users_df, pd.DataFrame(), pd.DataFrame(), now_utc=now_utc)

    ordered_trips["trip_number"] = ordered_trips.groupby("user_id").cumcount() + 1

    trip_summary_df = (
        ordered_trips.groupby("user_id")
        .agg(
            total_trips=("id", "count"),
            completed_trips=("is_completed", "sum"),
            first_trip_at=("created_at", "min"),
            last_trip_at=("created_at", "max"),
        )
        .reset_index()
        .rename(columns={"user_id": "id"})
    )

    third_trip_df = (
        ordered_trips[ordered_trips["trip_number"] == 3][["user_id", "created_at"]]
        .rename(columns={"user_id": "id", "created_at": "third_trip_at"})
    )

    user_activity_df = users_df.merge(trip_summary_df, on="id", how="left").merge(
        third_trip_df, on="id", how="left"
    )
    for col in ["total_trips", "completed_trips"]:
        user_activity_df[col] = pd.to_numeric(user_activity_df[col], errors="coerce").fillna(0).astype(int)

    raw_days = (user_activity_df["first_trip_at"] - user_activity_df["registration_at"]).dt.total_seconds() / 86400
    user_activity_df["days_to_first_trip"] = raw_days.where(
        user_activity_df["first_trip_at"].notna()
    ).clip(lower=0)
    user_activity_df["days_to_first_trip_bucket"] = (
        np.floor(user_activity_df["days_to_first_trip"]).where(user_activity_df["days_to_first_trip"].notna()).astype("Int64")
    )

    total_registered = int(len(user_activity_df))
    users_1_trip = int((user_activity_df["total_trips"] >= 1).sum())
    users_3_trips = int((user_activity_df["total_trips"] >= 3).sum())
    users_7_day = int((user_activity_df["days_to_first_trip"] <= 7).sum())

    activation_breakdown_df = pd.DataFrame(
        [
            {
                "milestone": "1+ Trips",
                "users": users_1_trip,
                "rate": (users_1_trip / total_registered * 100) if total_registered > 0 else 0,
            },
            {
                "milestone": "3+ Trips",
                "users": users_3_trips,
                "rate": (users_3_trips / total_registered * 100) if total_registered > 0 else 0,
            },
            {
                "milestone": "1st Trip in 7 Days",
                "users": users_7_day,
                "rate": (users_7_day / total_registered * 100) if total_registered > 0 else 0,
            },
        ]
    )

    cohort_summary_df = (
        user_activity_df.groupby("registration_week")
        .agg(
            total_users=("id", "count"),
            users_1_trip=("total_trips", lambda s: int((s >= 1).sum())),
            users_3_trips=("total_trips", lambda s: int((s >= 3).sum())),
            users_7_day=("days_to_first_trip", lambda s: int((s <= 7).sum())),
        )
        .reset_index()
        .sort_values("registration_week", ascending=False)
    )

    for numerator_col, rate_col in [
        ("users_1_trip", "rate_1_trip"),
        ("users_3_trips", "rate_3_trips"),
        ("users_7_day", "rate_7_day"),
    ]:
        cohort_summary_df[rate_col] = np.where(
            cohort_summary_df["total_users"] > 0,
            cohort_summary_df[numerator_col] / cohort_summary_df["total_users"] * 100,
            0,
        )

    activation_dist_df = (
        user_activity_df.dropna(subset=["days_to_first_trip_bucket"])
        .groupby("days_to_first_trip_bucket")["id"]
        .nunique()
        .reset_index(name="users")
        .rename(columns={"days_to_first_trip_bucket": "days_to_first_trip"})
        .sort_values("days_to_first_trip")
        .head(30)
    )

    activity_df = ordered_trips.merge(
        users_df[["id", "registration_at", "registration_week"]],
        left_on="user_id",
        right_on="id",
        how="inner",
    )
    activity_df = activity_df[activity_df["created_at"] >= activity_df["registration_at"]].copy()
    activity_df["activity_week"] = to_week_start(activity_df["created_at"])
    activity_df["weeks_since_registration"] = (
        (activity_df["activity_week"] - activity_df["registration_week"]).dt.days // 7
    ).astype(int)
    activity_df = activity_df[activity_df["weeks_since_registration"] >= 0]

    cohort_sizes = users_df.groupby("registration_week")["id"].nunique().sort_index()
    weekly_retention_df = pd.DataFrame()

    if not activity_df.empty:
        weekly_retention_df = (
            activity_df[["user_id", "registration_week", "weeks_since_registration"]]
            .drop_duplicates()
            .groupby(["registration_week", "weeks_since_registration"])["user_id"]
            .nunique()
            .reset_index(name="active_users")
        )
        weekly_retention_df["cohort_size"] = weekly_retention_df["registration_week"].map(cohort_sizes)
        weekly_retention_df["retention_rate"] = np.where(
            weekly_retention_df["cohort_size"] > 0,
            weekly_retention_df["active_users"] / weekly_retention_df["cohort_size"] * 100,
            0,
        )

    current_week_start = to_week_start(pd.Series([now_utc])).iloc[0]

    def weighted_retention_for_week(week_number: int):
        eligible_cohorts = cohort_sizes.index[
            cohort_sizes.index <= (current_week_start - pd.Timedelta(days=7 * week_number))
        ]
        eligible_users = int(cohort_sizes.reindex(eligible_cohorts).fillna(0).sum())
        if eligible_users == 0 or weekly_retention_df.empty:
            return 0.0, eligible_users

        week_activity = (
            weekly_retention_df[weekly_retention_df["weeks_since_registration"] == week_number]
            .set_index("registration_week")["active_users"]
        )
        retained_users = int(week_activity.reindex(eligible_cohorts).fillna(0).sum())
        rate = retained_users / eligible_users * 100 if eligible_users > 0 else 0
        return round(rate, 1), eligible_users

    week_1_retention, week_1_eligible_users = weighted_retention_for_week(1)
    week_4_retention, week_4_eligible_users = weighted_retention_for_week(4)

    median_days_to_first_trip = (
        round(float(user_activity_df["days_to_first_trip"].dropna().median()), 1)
        if user_activity_df["days_to_first_trip"].notna().any()
        else None
    )

    summary = {
        "total_registered": total_registered,
        "activation_1_trip_users": users_1_trip,
        "activation_1_trip_rate": round((users_1_trip / total_registered * 100) if total_registered > 0 else 0, 1),
        "activation_3_trip_users": users_3_trips,
        "activation_3_trip_rate": round((users_3_trips / total_registered * 100) if total_registered > 0 else 0, 1),
        "activation_7_day_users": users_7_day,
        "activation_7_day_rate": round((users_7_day / total_registered * 100) if total_registered > 0 else 0, 1),
        "week_1_retention": week_1_retention,
        "week_4_retention": week_4_retention,
        "week_1_eligible_users": week_1_eligible_users,
        "week_4_eligible_users": week_4_eligible_users,
        "median_days_to_first_trip": median_days_to_first_trip,
    }

    return summary, activation_breakdown_df, cohort_summary_df, weekly_retention_df, activation_dist_df


def build_subscription_segment_behavior_summary(
    profiles_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    events_df: pd.DataFrame,
    lookback_days: int = 30,
    now_utc: Optional[pd.Timestamp] = None,
):
    """Compare free vs premium behavior over a configurable lookback window."""
    empty_summary = pd.DataFrame(
        columns=[
            "Segment",
            "Users",
            "Active Users",
            "Active Rate",
            "Avg Trips/User",
            "Avg Completed Trips/User",
            "3+ Trip Users",
            "3+ Trip Rate",
            "Tracked Events",
            "Avg Tracked Events/User",
            "Paywall Open Users",
            "Purchase Start Users",
        ]
    )
    if profiles_df.empty:
        return empty_summary

    now_utc = now_utc or pd.Timestamp.now(tz="UTC")
    cutoff_dt = now_utc - pd.Timedelta(days=lookback_days)
    users_df = _ensure_columns(profiles_df, ["id", "subscription_segment"])[["id", "subscription_segment"]].copy()

    trips_df = _ensure_columns(trips_df, ["id", "user_id", "created_at", "is_completed", "distance"])
    if trips_df.empty:
        trip_user_summary = pd.DataFrame(columns=["id", "trips", "completed_trips", "distance"])
    else:
        trip_frame = trips_df.copy()
        trip_frame["created_at"] = pd.to_datetime(trip_frame["created_at"], errors="coerce", utc=True)
        recent_trips_df = trip_frame[trip_frame["created_at"] >= cutoff_dt].copy()

        if recent_trips_df.empty:
            trip_user_summary = pd.DataFrame(columns=["id", "trips", "completed_trips", "distance"])
        else:
            trip_user_summary = (
                recent_trips_df.groupby("user_id")
                .agg(
                    trips=("id", "count"),
                    completed_trips=("is_completed", "sum"),
                    distance=("distance", "sum"),
                )
                .reset_index()
                .rename(columns={"user_id": "id"})
            )

    events_df = _ensure_columns(events_df, ["id", "user_id", "event_name"])
    if events_df.empty:
        event_user_summary = pd.DataFrame(columns=["id", "tracked_events", "paywall_opens", "purchase_starts"])
    else:
        event_analysis_df = events_df.copy()
        event_analysis_df["event_name_norm"] = (
            event_analysis_df["event_name"].fillna("").astype(str).str.strip().str.lower()
        )
        event_user_summary = (
            event_analysis_df.groupby("user_id")
            .agg(
                tracked_events=("id", "count"),
                paywall_opens=("event_name_norm", lambda s: s.str.contains("paywall_opened", na=False).sum()),
                purchase_starts=("event_name_norm", lambda s: s.str.contains("purchase_started", na=False).sum()),
            )
            .reset_index()
            .rename(columns={"user_id": "id"})
        )

    behavior_df = users_df.merge(trip_user_summary, on="id", how="left").merge(
        event_user_summary, on="id", how="left"
    )

    for col in ["trips", "completed_trips", "distance", "tracked_events", "paywall_opens", "purchase_starts"]:
        behavior_df[col] = pd.to_numeric(behavior_df[col], errors="coerce").fillna(0)

    behavior_df["is_active"] = behavior_df["trips"] > 0
    behavior_df["is_power_user"] = behavior_df["trips"] >= 3
    behavior_df["has_paywall_open"] = behavior_df["paywall_opens"] > 0
    behavior_df["has_purchase_start"] = behavior_df["purchase_starts"] > 0

    summary_df = (
        behavior_df.groupby("subscription_segment")
        .agg(
            Users=("id", "count"),
            Active_Users=("is_active", "sum"),
            Avg_Trips_Per_User=("trips", "mean"),
            Avg_Completed_Trips_Per_User=("completed_trips", "mean"),
            Three_Plus_Trip_Users=("is_power_user", "sum"),
            Tracked_Events=("tracked_events", "sum"),
            Avg_Tracked_Events_Per_User=("tracked_events", "mean"),
            Paywall_Open_Users=("has_paywall_open", "sum"),
            Purchase_Start_Users=("has_purchase_start", "sum"),
        )
        .reset_index()
        .rename(columns={"subscription_segment": "Segment"})
    )

    summary_df["Active Rate"] = np.where(
        summary_df["Users"] > 0,
        summary_df["Active_Users"] / summary_df["Users"] * 100,
        0,
    )
    summary_df["3+ Trip Rate"] = np.where(
        summary_df["Users"] > 0,
        summary_df["Three_Plus_Trip_Users"] / summary_df["Users"] * 100,
        0,
    )

    return summary_df[
        [
            "Segment",
            "Users",
            "Active_Users",
            "Active Rate",
            "Avg_Trips_Per_User",
            "Avg_Completed_Trips_Per_User",
            "Three_Plus_Trip_Users",
            "3+ Trip Rate",
            "Tracked_Events",
            "Avg_Tracked_Events_Per_User",
            "Paywall_Open_Users",
            "Purchase_Start_Users",
        ]
    ].rename(
        columns={
            "Active_Users": "Active Users",
            "Avg_Trips_Per_User": "Avg Trips/User",
            "Avg_Completed_Trips_Per_User": "Avg Completed Trips/User",
            "Three_Plus_Trip_Users": "3+ Trip Users",
            "Tracked_Events": "Tracked Events",
            "Avg_Tracked_Events_Per_User": "Avg Tracked Events/User",
            "Paywall_Open_Users": "Paywall Open Users",
            "Purchase_Start_Users": "Purchase Start Users",
        }
    )
