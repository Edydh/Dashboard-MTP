"""
Shared analytics transforms used by dashboard views.
"""

import numpy as np
import pandas as pd

from dashboard_data import normalize_subscription_tier

UPGRADE_STAGE_PATTERNS = {
    "Paywall Opened": ["paywall_opened", "paywall_viewed", "paywall_shown", "paywall_displayed"],
    "Purchase Started": ["purchase_started", "checkout_started"],
    "Purchase Completed": [
        "purchase_completed",
        "purchase_complete",
        "purchase_succeeded",
        "subscription_purchased",
    ],
}

REVENUECAT_STAGE_MAP = {
    "INITIAL_PURCHASE": "Purchase",
    "NON_RENEWING_PURCHASE": "Purchase",
    "TRIAL_STARTED": "Trial",
    "RENEWAL": "Renewal",
    "CANCELLATION": "Cancellation",
    "UNCANCELLATION": "Uncancellation",
    "EXPIRATION": "Expiration",
    "BILLING_ISSUE": "Billing Issue",
    "PRODUCT_CHANGE": "Product Change",
    "REFUND": "Refund",
    "SUBSCRIPTION_PAUSED": "Paused",
    "TEMPORARY_ENTITLEMENT_GRANT": "Entitlement Grant",
    "TRANSFER": "Transfer",
}


def _ensure_columns(df: pd.DataFrame, columns) -> pd.DataFrame:
    result = df.copy() if not df.empty else pd.DataFrame()
    for col in columns:
        if col not in result.columns:
            result[col] = None
    return result


def classify_revenuecat_lifecycle_stage(event_type) -> str:
    """Map RevenueCat event types into lifecycle stages for dashboard display."""
    normalized_type = str(event_type or "").strip().upper()
    return REVENUECAT_STAGE_MAP.get(normalized_type, "Other")


def build_revenuecat_lifecycle_summary(events_df: pd.DataFrame, app_user_id=None):
    """Build a user-level RevenueCat lifecycle timeline and conservative state summary."""
    timeline_columns = [
        "event_time",
        "stage",
        "event_type",
        "app_user_id",
        "product_id",
        "entitlement_display",
        "store",
        "environment",
        "transaction_id",
        "revenuecat_event_id",
    ]
    empty_timeline = pd.DataFrame(columns=timeline_columns)
    empty_summary = {
        "events": 0,
        "app_users": 0,
        "purchase_events": 0,
        "renewal_events": 0,
        "cancellation_events": 0,
        "expiration_events": 0,
        "refund_events": 0,
        "billing_issue_events": 0,
        "latest_event_type": None,
        "latest_event_at": None,
        "latest_product_id": None,
        "latest_environment": None,
        "latest_store": None,
        "current_state": "No events",
        "state_detail": "No RevenueCat lifecycle events matched the current selection.",
    }

    if events_df.empty:
        return empty_timeline, empty_summary

    lifecycle_df = _ensure_columns(
        events_df,
        [
            "event_timestamp",
            "created_at",
            "event_type",
            "app_user_id",
            "product_id",
            "entitlement_display",
            "store",
            "environment",
            "transaction_id",
            "revenuecat_event_id",
        ],
    ).copy()

    lifecycle_df["app_user_id"] = lifecycle_df["app_user_id"].astype(str)
    lifecycle_df = lifecycle_df[lifecycle_df["app_user_id"].str.strip().ne("")]
    if app_user_id:
        lifecycle_df = lifecycle_df[lifecycle_df["app_user_id"].eq(str(app_user_id))]

    if lifecycle_df.empty:
        return empty_timeline, empty_summary

    lifecycle_df["event_time"] = pd.to_datetime(
        lifecycle_df["event_timestamp"], errors="coerce", utc=True
    ).combine_first(pd.to_datetime(lifecycle_df["created_at"], errors="coerce", utc=True))
    lifecycle_df["event_type"] = lifecycle_df["event_type"].fillna("UNKNOWN").astype(str).str.upper()
    lifecycle_df["stage"] = lifecycle_df["event_type"].apply(classify_revenuecat_lifecycle_stage)
    lifecycle_df = lifecycle_df.sort_values("event_time", ascending=True, na_position="last")

    timeline_df = lifecycle_df[timeline_columns].reset_index(drop=True)

    stage_counts = timeline_df["stage"].value_counts()
    latest_row = timeline_df.dropna(subset=["event_time"]).tail(1)
    if latest_row.empty:
        latest_row = timeline_df.tail(1)
    latest = latest_row.iloc[0]

    purchase_events = int(stage_counts.get("Purchase", 0))
    renewal_events = int(stage_counts.get("Renewal", 0))
    cancellation_events = int(stage_counts.get("Cancellation", 0))
    expiration_events = int(stage_counts.get("Expiration", 0))
    refund_events = int(stage_counts.get("Refund", 0))
    billing_issue_events = int(stage_counts.get("Billing Issue", 0))

    if refund_events > 0:
        current_state = "Refunded"
        state_detail = "A refund event is present. Verify entitlement state before treating this user as active."
    elif expiration_events > 0:
        current_state = "Expired"
        state_detail = "An expiration event is present after the purchase lifecycle began."
    elif cancellation_events > 0:
        current_state = "Cancelled"
        state_detail = "Cancellation was recorded; access may remain active until expiration."
    elif purchase_events > 0 or renewal_events > 0:
        current_state = "Active Signal"
        state_detail = "Purchase or renewal events are present and no cancellation/expiration/refund was found."
    else:
        current_state = "No Purchase"
        state_detail = "No purchase, renewal, cancellation, expiration, or refund signal was found."

    summary = {
        "events": int(len(timeline_df)),
        "app_users": int(timeline_df["app_user_id"].nunique()),
        "purchase_events": purchase_events,
        "renewal_events": renewal_events,
        "cancellation_events": cancellation_events,
        "expiration_events": expiration_events,
        "refund_events": refund_events,
        "billing_issue_events": billing_issue_events,
        "latest_event_type": latest.get("event_type"),
        "latest_event_at": latest.get("event_time"),
        "latest_product_id": latest.get("product_id"),
        "latest_environment": latest.get("environment"),
        "latest_store": latest.get("store"),
        "current_state": current_state,
        "state_detail": state_detail,
    }
    return timeline_df, summary


def _prepare_upgrade_analysis_df(events_df: pd.DataFrame) -> pd.DataFrame:
    analysis_df = _ensure_columns(
        events_df,
        ["id", "user_id", "event_name", "feature_display", "subscription_tier"],
    )
    if analysis_df.empty:
        return analysis_df

    analysis_df["user_id"] = analysis_df["user_id"].astype(str)
    analysis_df["event_name_norm"] = (
        analysis_df["event_name"].fillna("").astype(str).str.strip().str.lower()
    )
    analysis_df["feature_display"] = (
        analysis_df["feature_display"].fillna("Unknown").replace("", "Unknown")
    )
    analysis_df["subscription_tier_norm"] = normalize_subscription_tier(
        analysis_df["subscription_tier"]
    )
    return analysis_df


def _build_stage_mask(analysis_df: pd.DataFrame, stage_name: str) -> pd.Series:
    patterns = UPGRADE_STAGE_PATTERNS[stage_name]
    return analysis_df["event_name_norm"].apply(
        lambda value: any(pattern in value for pattern in patterns)
    )


def build_upgrade_purchase_summary(events_df: pd.DataFrame):
    """Summarize purchase-start and purchase-complete users from upgrade signals."""
    empty_summary = {
        "purchase_start_users": 0,
        "purchase_start_events": 0,
        "purchase_complete_event_users": 0,
        "purchase_complete_events": 0,
        "inferred_purchase_complete_users": 0,
        "observed_purchase_complete_users": 0,
    }

    analysis_df = _prepare_upgrade_analysis_df(events_df)
    if analysis_df.empty:
        return empty_summary

    purchase_start_mask = _build_stage_mask(analysis_df, "Purchase Started")
    purchase_complete_mask = _build_stage_mask(analysis_df, "Purchase Completed")

    purchase_start_user_ids = set(analysis_df.loc[purchase_start_mask, "user_id"])
    purchase_complete_event_user_ids = set(analysis_df.loc[purchase_complete_mask, "user_id"])
    paid_tier_user_ids = set(
        analysis_df.loc[analysis_df["subscription_tier_norm"].ne("free"), "user_id"]
    )
    inferred_purchase_complete_user_ids = (
        paid_tier_user_ids & purchase_start_user_ids
    ) - purchase_complete_event_user_ids
    observed_purchase_complete_user_ids = (
        purchase_complete_event_user_ids | inferred_purchase_complete_user_ids
    )

    return {
        "purchase_start_users": len(purchase_start_user_ids),
        "purchase_start_events": int(purchase_start_mask.sum()),
        "purchase_complete_event_users": len(purchase_complete_event_user_ids),
        "purchase_complete_events": int(purchase_complete_mask.sum()),
        "inferred_purchase_complete_users": len(inferred_purchase_complete_user_ids),
        "observed_purchase_complete_users": len(observed_purchase_complete_user_ids),
    }


def build_upgrade_conversion_funnel(events_df: pd.DataFrame):
    """Build a paywall-to-purchase funnel from normalized upgrade events."""
    empty_funnel = pd.DataFrame(
        columns=["Stage", "Users", "Events", "Conversion from Previous", "Conversion from Paywall"]
    )
    empty_paywall = pd.DataFrame(columns=["Feature", "Paywall Opens", "Users"])

    if events_df.empty:
        return empty_funnel, empty_paywall, False

    analysis_df = _prepare_upgrade_analysis_df(events_df)
    paywall_mask = _build_stage_mask(analysis_df, "Paywall Opened")
    purchase_start_mask = _build_stage_mask(analysis_df, "Purchase Started")
    purchase_complete_mask = _build_stage_mask(analysis_df, "Purchase Completed")

    paywall_users = set(analysis_df.loc[paywall_mask, "user_id"])
    purchase_started_users = set(analysis_df.loc[purchase_start_mask, "user_id"]) & paywall_users
    purchase_completed_users = (
        set(analysis_df.loc[purchase_complete_mask, "user_id"]) & paywall_users
    )

    funnel_rows = [
        {
            "Stage": "Paywall Opened",
            "Users": len(paywall_users),
            "Events": int(paywall_mask.sum()),
        },
        {
            "Stage": "Purchase Started",
            "Users": len(purchase_started_users),
            "Events": int((purchase_start_mask & analysis_df["user_id"].isin(paywall_users)).sum()),
        },
        {
            "Stage": "Purchase Completed",
            "Users": len(purchase_completed_users),
            "Events": int(
                (purchase_complete_mask & analysis_df["user_id"].isin(paywall_users)).sum()
            ),
        },
    ]

    funnel_df = pd.DataFrame(funnel_rows)
    previous_users = None
    paywall_user_count = funnel_df.iloc[0]["Users"] if not funnel_df.empty else 0
    conversion_from_previous = []
    conversion_from_paywall = []

    for _, row in funnel_df.iterrows():
        current_users = row["Users"]
        if previous_users in (None, 0):
            conversion_from_previous.append(100.0 if current_users > 0 else 0.0)
        else:
            conversion_from_previous.append(current_users / previous_users * 100)

        if paywall_user_count > 0:
            conversion_from_paywall.append(current_users / paywall_user_count * 100)
        else:
            conversion_from_paywall.append(0.0)

        previous_users = current_users

    funnel_df["Conversion from Previous"] = conversion_from_previous
    funnel_df["Conversion from Paywall"] = conversion_from_paywall

    paywall_feature_df = (
        analysis_df[paywall_mask]
        .groupby("feature_display")
        .agg(
            **{
                "Paywall Opens": ("id", "count"),
                "Users": ("user_id", "nunique"),
            }
        )
        .reset_index()
        .rename(columns={"feature_display": "Feature"})
        .sort_values(["Users", "Paywall Opens"], ascending=[False, False])
        .head(10)
    )

    has_completed_purchase_events = bool(
        funnel_df.loc[funnel_df["Stage"] == "Purchase Completed", "Users"].max() > 0
    ) if not funnel_df.empty else False

    return funnel_df, paywall_feature_df, has_completed_purchase_events


def build_feature_usage_summary(events_df: pd.DataFrame):
    """Summarize tracked feature usage and split it by free vs premium users."""
    empty_summary = pd.DataFrame(
        columns=[
            "Feature",
            "Events",
            "Unique Users",
            "Free Users",
            "Premium Users",
            "Free Events",
            "Premium Events",
            "Paywall Opens",
            "Purchase Starts",
            "Last Seen",
        ]
    )

    if events_df.empty:
        return empty_summary

    usage_df = events_df.copy()
    usage_df["event_name_norm"] = usage_df["event_name"].fillna("").astype(str).str.strip().str.lower()
    usage_df["feature_display"] = usage_df["feature_display"].fillna("Unknown").replace("", "Unknown")
    usage_df["subscription_segment"] = np.where(
        normalize_subscription_tier(usage_df["subscription_tier"]).eq("free"),
        "Free",
        "Premium",
    )

    feature_summary = (
        usage_df.groupby("feature_display")
        .agg(
            **{
                "Events": ("id", "count"),
                "Unique Users": ("user_id", "nunique"),
                "Paywall Opens": (
                    "event_name_norm",
                    lambda s: s.str.contains("paywall_opened", na=False).sum(),
                ),
                "Purchase Starts": (
                    "event_name_norm",
                    lambda s: s.str.contains("purchase_started", na=False).sum(),
                ),
                "Last Seen": ("event_ts", "max"),
            }
        )
        .reset_index()
        .rename(columns={"feature_display": "Feature"})
    )

    user_segment_counts = (
        usage_df[["feature_display", "user_id", "subscription_segment"]]
        .drop_duplicates()
        .groupby(["feature_display", "subscription_segment"])
        .size()
        .unstack(fill_value=0)
    )
    event_segment_counts = (
        usage_df.groupby(["feature_display", "subscription_segment"]).size().unstack(fill_value=0)
    )

    for segment in ["Free", "Premium"]:
        feature_summary[f"{segment} Users"] = feature_summary["Feature"].map(
            user_segment_counts.get(segment, pd.Series(dtype="int64"))
        ).fillna(0).astype(int)
        feature_summary[f"{segment} Events"] = feature_summary["Feature"].map(
            event_segment_counts.get(segment, pd.Series(dtype="int64"))
        ).fillna(0).astype(int)

    feature_summary["Last Seen"] = pd.to_datetime(
        feature_summary["Last Seen"], errors="coerce", utc=True
    )

    return feature_summary.sort_values(
        ["Unique Users", "Events"], ascending=[False, False]
    ).reset_index(drop=True)


def build_trip_purpose_analytics(trips_df: pd.DataFrame):
    """Normalize trip purpose records and classify them into broad categories."""
    purpose_df = _ensure_columns(
        trips_df,
        ["purpose", "user_id", "distance", "fuel_used", "reimbursement", "created_at"],
    )
    if purpose_df.empty:
        return pd.DataFrame()

    purpose_df["created_at"] = pd.to_datetime(purpose_df["created_at"], errors="coerce", utc=True)
    purpose_df = purpose_df.dropna(subset=["created_at"]).copy()
    if purpose_df.empty:
        return pd.DataFrame()

    purpose_df["distance"] = pd.to_numeric(purpose_df["distance"], errors="coerce").fillna(0)
    purpose_df["fuel_used"] = pd.to_numeric(purpose_df["fuel_used"], errors="coerce").fillna(0)
    purpose_df["reimbursement"] = pd.to_numeric(
        purpose_df["reimbursement"], errors="coerce"
    ).fillna(0)
    purpose_df["purpose"] = purpose_df["purpose"].fillna("Not Specified")

    business_terms = ["business", "meeting", "client", "work", "office", "conference"]
    personal_terms = ["personal", "home", "shopping", "doctor", "family", "vacation"]
    service_terms = ["delivery", "pickup", "service"]

    def categorize_purpose(purpose):
        purpose_lower = str(purpose).strip().lower()
        if any(term in purpose_lower for term in business_terms):
            return "Business"
        if any(term in purpose_lower for term in personal_terms):
            return "Personal"
        if any(term in purpose_lower for term in service_terms):
            return "Service/Delivery"
        return "Other"

    purpose_df["purpose_category"] = purpose_df["purpose"].apply(categorize_purpose)
    return purpose_df


def build_fuel_efficiency_analytics(trips_df: pd.DataFrame, avg_fuel_price: float = 3.50):
    """Calculate trip-level fuel efficiency and cost metrics."""
    fuel_df = _ensure_columns(
        trips_df,
        ["distance", "fuel_used", "reimbursement", "user_id", "created_at", "purpose"],
    )
    if fuel_df.empty:
        return pd.DataFrame()

    fuel_df["created_at"] = pd.to_datetime(fuel_df["created_at"], errors="coerce", utc=True)
    fuel_df = fuel_df.dropna(subset=["created_at"]).copy()
    if fuel_df.empty:
        return pd.DataFrame()

    fuel_df["distance"] = pd.to_numeric(fuel_df["distance"], errors="coerce").fillna(0)
    fuel_df["fuel_used"] = pd.to_numeric(fuel_df["fuel_used"], errors="coerce").fillna(0)
    fuel_df["reimbursement"] = pd.to_numeric(fuel_df["reimbursement"], errors="coerce").fillna(0)
    fuel_df["mpg"] = np.where(
        fuel_df["fuel_used"] > 0,
        fuel_df["distance"] / fuel_df["fuel_used"],
        0,
    )
    fuel_df["fuel_cost"] = fuel_df["fuel_used"] * avg_fuel_price
    fuel_df["cost_per_mile"] = np.where(
        fuel_df["distance"] > 0,
        fuel_df["fuel_cost"] / fuel_df["distance"],
        0,
    )
    return fuel_df
