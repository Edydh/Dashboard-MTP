"""
Shared data-access and normalization helpers for dashboard views.
"""

import numpy as np
import pandas as pd
import streamlit as st

from supabase_client import Client

PROFILE_DATASET_COLUMNS = [
    "id",
    "full_name",
    "phone_number",
    "subscription_tier",
    "created_at",
    "subscription_tier_norm",
    "subscription_segment",
]

TRIP_ACTIVITY_DATASET_COLUMNS = [
    "id",
    "user_id",
    "created_at",
    "status",
    "mileage",
    "actual_distance",
    "distance",
    "status_norm",
    "is_completed",
]

TRIP_METRICS_DATASET_COLUMNS = [
    "id",
    "user_id",
    "created_at",
    "status",
    "status_norm",
    "is_completed",
    "mileage",
    "actual_distance",
    "distance",
    "start_time",
    "end_time",
    "duration",
    "purpose",
    "fuel_used",
    "reimbursement",
]


def normalize_subscription_tier(series: pd.Series) -> pd.Series:
    """Normalize subscription tiers so downstream comparisons are consistent."""
    normalized = series.fillna("free").astype(str).str.strip().str.lower()
    return normalized.replace({"": "free", "nan": "free", "none": "free", "null": "free"})


def to_week_start(values) -> pd.Series:
    """Return the Monday-start week bucket for a datetime-like series."""
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    return timestamps.dt.normalize() - pd.to_timedelta(timestamps.dt.weekday, unit="D")


def parse_mixed_timestamp_series(values, utc=True):
    """Parse timestamp columns that mix fractional and non-fractional ISO strings."""
    return pd.to_datetime(values, errors="coerce", utc=utc, format="mixed")


def normalize_trip_distance(actual_distance_values, mileage_values) -> pd.Series:
    """Use actual distance when present and otherwise fall back to mileage."""
    actual_distance = pd.to_numeric(actual_distance_values, errors="coerce")
    mileage = pd.to_numeric(mileage_values, errors="coerce")
    return actual_distance.fillna(mileage).fillna(0)


def calculate_trip_duration_minutes(start_values, end_values):
    """Calculate trip duration in minutes and leave unresolved/non-positive values empty."""
    start_times = parse_mixed_timestamp_series(start_values, utc=True)
    end_times = parse_mixed_timestamp_series(end_values, utc=True)
    duration_minutes = (end_times - start_times).dt.total_seconds() / 60
    return duration_minutes.where(duration_minutes > 0)


@st.cache_data(ttl=300)
def get_trips_dataframe(_supabase: Client, columns: str, created_at_gte=None):
    """Fetch all trips for the requested columns using pagination."""
    try:
        all_rows = []
        page_size = 1000
        start = 0

        while True:
            query = _supabase.table("trips").select(columns).order("created_at", desc=False)
            if created_at_gte is not None:
                query = query.gte("created_at", created_at_gte)

            batch = query.range(start, start + page_size - 1).execute().data or []
            if not batch:
                break

            all_rows.extend(batch)
            if len(batch) < page_size:
                break

            start += page_size

        return pd.DataFrame(all_rows)
    except Exception as exc:
        st.error(f"Error fetching trip data: {str(exc)}")
        return pd.DataFrame()


def _empty_profile_dataset() -> pd.DataFrame:
    return pd.DataFrame(columns=PROFILE_DATASET_COLUMNS)


@st.cache_data(ttl=300)
def get_profile_dataset(_supabase: Client):
    """Fetch profile context used across retention and segmentation views."""
    try:
        response = _supabase.table("profiles").select(
            "id, full_name, phone_number, subscription_tier, created_at"
        ).execute()

        profiles_df = pd.DataFrame(response.data or [])
        expected_columns = [
            "id",
            "full_name",
            "phone_number",
            "subscription_tier",
            "created_at",
        ]

        if profiles_df.empty:
            return _empty_profile_dataset()

        for col in expected_columns:
            if col not in profiles_df.columns:
                profiles_df[col] = None

        profiles_df["id"] = profiles_df["id"].astype(str)
        profiles_df["created_at"] = pd.to_datetime(
            profiles_df["created_at"], errors="coerce", utc=True
        )
        profiles_df["subscription_tier_norm"] = normalize_subscription_tier(
            profiles_df["subscription_tier"]
        )
        profiles_df["subscription_segment"] = np.where(
            profiles_df["subscription_tier_norm"].eq("free"),
            "Free",
            "Premium",
        )

        return profiles_df[PROFILE_DATASET_COLUMNS]
    except Exception as exc:
        st.error(f"Error fetching profiles: {str(exc)}")
        return _empty_profile_dataset()


def _empty_trip_activity_dataset() -> pd.DataFrame:
    return pd.DataFrame(columns=TRIP_ACTIVITY_DATASET_COLUMNS)


def _empty_trip_metrics_dataset() -> pd.DataFrame:
    return pd.DataFrame(columns=TRIP_METRICS_DATASET_COLUMNS)


def prepare_trip_metrics_dataset(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the trip dataset used across summary and activity analytics."""
    expected_columns = [
        "id",
        "user_id",
        "created_at",
        "status",
        "mileage",
        "actual_distance",
        "start_time",
        "end_time",
        "purpose",
        "fuel_used",
        "reimbursement",
    ]

    if trips_df.empty:
        return _empty_trip_metrics_dataset()

    prepared_df = trips_df.copy()
    for col in expected_columns:
        if col not in prepared_df.columns:
            prepared_df[col] = None

    prepared_df["id"] = prepared_df["id"].astype(str)
    prepared_df["user_id"] = prepared_df["user_id"].astype(str)
    prepared_df["created_at"] = pd.to_datetime(
        prepared_df["created_at"], errors="coerce", utc=True
    )
    prepared_df["start_time"] = parse_mixed_timestamp_series(prepared_df["start_time"], utc=True)
    prepared_df["end_time"] = parse_mixed_timestamp_series(prepared_df["end_time"], utc=True)
    prepared_df = prepared_df.dropna(subset=["created_at"]).copy()

    if prepared_df.empty:
        return _empty_trip_metrics_dataset()

    prepared_df["distance"] = normalize_trip_distance(
        prepared_df["actual_distance"],
        prepared_df["mileage"],
    )
    prepared_df["duration"] = calculate_trip_duration_minutes(
        prepared_df["start_time"],
        prepared_df["end_time"],
    )
    prepared_df["status_norm"] = (
        prepared_df["status"].fillna("").astype(str).str.strip().str.lower()
    )
    prepared_df["is_completed"] = prepared_df["status_norm"].isin({"completed", "complated"})
    prepared_df["fuel_used"] = pd.to_numeric(prepared_df["fuel_used"], errors="coerce").fillna(0)
    prepared_df["reimbursement"] = pd.to_numeric(
        prepared_df["reimbursement"], errors="coerce"
    ).fillna(0)

    return prepared_df[TRIP_METRICS_DATASET_COLUMNS]


@st.cache_data(ttl=300)
def get_trip_metrics_dataset(_supabase: Client, created_at_gte=None):
    """Fetch and normalize the trip dataset used across analytics views."""
    trips_df = get_trips_dataframe(
        _supabase,
        (
            "id, user_id, created_at, status, mileage, actual_distance, "
            "start_time, end_time, purpose, fuel_used, reimbursement"
        ),
        created_at_gte=created_at_gte,
    )
    return prepare_trip_metrics_dataset(trips_df)


@st.cache_data(ttl=300)
def get_trip_activity_dataset(_supabase: Client):
    """Fetch a paginated trip activity dataset for retention and segmentation analysis."""
    trips_df = get_trip_metrics_dataset(_supabase)
    if trips_df.empty:
        return _empty_trip_activity_dataset()
    return trips_df[TRIP_ACTIVITY_DATASET_COLUMNS].copy()
