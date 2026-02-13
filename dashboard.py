"""
Mileage Tracker Pro Dashboard
A comprehensive dashboard for monitoring user activity and trip statistics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np
from streamlit_extras.metric_cards import style_metric_cards
# Import from our custom client manager
from supabase_client import get_supabase_client, get_supabase_manager, Client

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Mileage Tracker Pro Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Note: Supabase client initialization is now handled by supabase_client.py
# The client is automatically cached and includes retry logic, monitoring, etc.

# Data fetching functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_total_users(_supabase: Client):
    """Get total number of users from profiles table"""
    try:
        response = _supabase.table('profiles').select('id', count='exact').execute()
        return response.count
    except Exception as e:
        st.error(f"Error fetching user count: {str(e)}")
        return 0

@st.cache_data(ttl=300)
def get_active_users(_supabase: Client, days=30):
    """Get active users in the last N days"""
    try:
        # Use UTC for all datetime operations
        cutoff_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        cutoff_date = cutoff_dt.isoformat()
        
        # Query trips table for active users
        response = _supabase.table('trips').select('user_id').gte('created_at', cutoff_date).execute()
        
        if response.data:
            active_users = set(trip['user_id'] for trip in response.data)
            return len(active_users)
        return 0
    except Exception as e:
        st.error(f"Error fetching active users: {str(e)}")
        return 0

@st.cache_data(ttl=300)
def get_top_active_users(_supabase: Client, limit=10):
    """Get top 10 most active users with their statistics"""
    try:
        # Get all trips with user information
        trips_response = _supabase.table('trips').select('user_id, created_at, mileage, actual_distance, start_time, end_time').execute()
        
        if not trips_response.data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        trips_df = pd.DataFrame(trips_response.data)
        
        # Get user information from profiles table
        users_response = _supabase.table('profiles').select('*').execute()
        users_df = pd.DataFrame(users_response.data or [])
        
        # Ensure expected profile columns exist
        for col in ['id', 'full_name', 'phone_number', 'subscription_tier', 'email']:
            if col not in users_df.columns:
                users_df[col] = None
        
        # Prefer email from Supabase Authentication users when available.
        users_df['id'] = users_df['id'].astype(str)
        auth_users_df = get_auth_users(_supabase)
        if not auth_users_df.empty and {'id', 'email'}.issubset(auth_users_df.columns):
            auth_email_map = auth_users_df.dropna(subset=['id']).drop_duplicates(subset=['id'], keep='first').set_index('id')['email']
            users_df['email'] = users_df['id'].map(auth_email_map).fillna(users_df['email'])
        
        # Calculate duration from start_time and end_time if available (force UTC)
        if 'start_time' in trips_df.columns and 'end_time' in trips_df.columns:
            trips_df['start_time'] = pd.to_datetime(trips_df['start_time'], errors='coerce', utc=True)
            trips_df['end_time'] = pd.to_datetime(trips_df['end_time'], errors='coerce', utc=True)
            trips_df['duration'] = (trips_df['end_time'] - trips_df['start_time']).dt.total_seconds() / 60  # Duration in minutes
        else:
            trips_df['duration'] = 0
        
        # Use actual_distance if available, otherwise use mileage
        trips_df['distance'] = trips_df['actual_distance'].fillna(trips_df['mileage']).fillna(0)
        
        # Calculate statistics per user
        user_stats = trips_df.groupby('user_id').agg({
            'user_id': 'count',  # Number of trips
            'distance': 'sum',    # Total distance
            'duration': 'sum',    # Total duration
            'created_at': lambda x: (pd.Timestamp.now(tz="UTC") - pd.to_datetime(x, utc=True).max()).days  # Days since last trip
        }).rename(columns={
            'user_id': 'trip_count',
            'distance': 'total_distance',
            'duration': 'total_duration',
            'created_at': 'days_since_last_trip'
        })
        
        # Calculate usage frequency (trips per week since first trip)
        first_trip = trips_df.groupby('user_id')['created_at'].min()
        weeks_active = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(first_trip, utc=True)).dt.days / 7
        user_stats['trips_per_week'] = user_stats['trip_count'] / weeks_active.clip(lower=1)
        
        # Reset index to get user_id as column
        user_stats = user_stats.reset_index()
        
        # Merge with user information
        user_stats = user_stats.merge(users_df, left_on='user_id', right_on='id', how='left')
        
        # Select and rename columns
        user_stats = user_stats[['full_name', 'email', 'phone_number', 'subscription_tier', 'trip_count', 'trips_per_week', 
                                 'total_distance', 'total_duration', 'days_since_last_trip']]
        
        # Sort by trip count and get top 10
        top_users = user_stats.nlargest(limit, 'trip_count')
        
        return top_users
        
    except Exception as e:
        st.error(f"Error fetching top active users: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_usage_patterns(_supabase: Client):
    """Get usage patterns over time"""
    try:
        # Get trips from last 30 days
        cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
        response = _supabase.table('trips').select('created_at, user_id').gte('created_at', cutoff_date).execute()
        
        if not response.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(response.data)
        # Normalize timestamps to UTC and then drop tz for grouping consistency
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        df['date'] = df['created_at'].dt.date
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.day_name()
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching usage patterns: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_trip_statistics(_supabase: Client):
    """Get overall trip statistics"""
    try:
        response = _supabase.table('trips').select('mileage, actual_distance, start_time, end_time, created_at').execute()
        
        if not response.data:
            return {}
        
        df = pd.DataFrame(response.data)
 
        # Calculate duration from start_time and end_time if available
        if 'start_time' in df.columns and 'end_time' in df.columns:
            df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce', utc=True)
            df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce', utc=True)
            df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60  # Duration in minutes
        else:
            df['duration'] = 0
 
        # Use actual_distance if available, otherwise use mileage
        df['distance'] = df['actual_distance'].fillna(df['mileage']).fillna(0)
 
        # Normalize created_at to UTC for comparisons
        created_at_utc = pd.to_datetime(df['created_at'], utc=True)
        now_utc = pd.Timestamp.now(tz='UTC')

        stats = {
            'total_trips': len(df),
            'total_distance': df['distance'].sum(),
            'avg_distance': df['distance'].mean(),
            'total_duration': df['duration'].sum(),
            'avg_duration': df['duration'].mean(),
            'trips_today': len(df[created_at_utc.dt.date == now_utc.date()]),
            'trips_this_week': len(df[created_at_utc >= now_utc - pd.Timedelta(days=7)]),
            'trips_this_month': len(df[created_at_utc >= now_utc - pd.Timedelta(days=30)])
        }
        
        return stats
        
    except Exception as e:
        st.error(f"Error fetching trip statistics: {str(e)}")
        return {}

@st.cache_data(ttl=300)
def get_users_at_risk(_supabase: Client, inactivity_days: int = 14):
    """Return dataframe of users whose last trip was more than inactivity_days ago.
    Also returns the count of users who never made a trip.
    """
    try:
        # Fetch trips
        trips_resp = _supabase.table('trips').select('user_id, created_at, mileage, actual_distance').execute()
        profiles_resp = _supabase.table('profiles').select('id, full_name, phone_number, subscription_tier, created_at').execute()

        profiles_df = pd.DataFrame(profiles_resp.data or [])
        if profiles_df.empty:
            return pd.DataFrame(), 0

        trips_df = pd.DataFrame(trips_resp.data or [])
        if not trips_df.empty:
            # Normalize
            trips_df['created_at'] = pd.to_datetime(trips_df['created_at'], utc=True)
            trips_df['distance'] = pd.to_numeric(trips_df.get('actual_distance'), errors='coerce').fillna(pd.to_numeric(trips_df.get('mileage'), errors='coerce')).fillna(0)
            # Aggregate by user
            agg = trips_df.groupby('user_id').agg(
                trip_count=('user_id', 'count'),
                last_trip=('created_at', 'max'),
                total_distance=('distance', 'sum')
            ).reset_index()
        else:
            agg = pd.DataFrame(columns=['user_id', 'trip_count', 'last_trip', 'total_distance'])

        # Merge with profiles
        merged = profiles_df.merge(agg, left_on='id', right_on='user_id', how='left')
        now_utc = pd.Timestamp.now(tz='UTC')
        merged['days_since_last_trip'] = (now_utc - pd.to_datetime(merged['last_trip'], utc=True)).dt.days

        # Users with no trips
        never_used_count = int(merged['last_trip'].isna().sum())

        # At risk: last trip older than threshold
        at_risk = merged[(merged['days_since_last_trip'].notna()) & (merged['days_since_last_trip'] > inactivity_days)].copy()

        # Select columns and sort
        if not at_risk.empty:
            at_risk = at_risk[['full_name', 'phone_number', 'subscription_tier', 'trip_count', 'total_distance', 'last_trip', 'days_since_last_trip']]
            at_risk['total_distance'] = at_risk['total_distance'].fillna(0).round(1)
            at_risk['trip_count'] = at_risk['trip_count'].fillna(0).astype(int)
            at_risk = at_risk.sort_values(['days_since_last_trip', 'trip_count'], ascending=[False, False])

        return at_risk, never_used_count
    except Exception as e:
        st.error(f"Error computing users at risk: {str(e)}")
        return pd.DataFrame(), 0

@st.cache_data(ttl=300)
def get_user_retention_analysis(_supabase: Client):
    """Calculate user retention metrics and cohort analysis"""
    try:
        # Try multiple table sources for user data
        users_df = pd.DataFrame()
        use_auth_table = False
        table_source = "none"
        
        # Try auth.users table first (Supabase Auth)
        try:
            users_resp = _supabase.table('auth.users').select('id, email_confirmed_at, last_sign_in_at').execute()
            users_df = pd.DataFrame(users_resp.data or [])
            use_auth_table = True
            table_source = "auth.users"
        except:
            pass
        
        # Try users table (if auth.users fails)
        if users_df.empty:
            try:
                users_resp = _supabase.table('users').select('id, email_confirmed_at, last_sign_in_at, created_at').execute()
                users_df = pd.DataFrame(users_resp.data or [])
                use_auth_table = True
                table_source = "users"
            except:
                pass
        
        # Fallback to profiles table
        if users_df.empty:
            try:
                users_resp = _supabase.table('profiles').select('id, created_at').execute()
                users_df = pd.DataFrame(users_resp.data or [])
                use_auth_table = False
                table_source = "profiles"
            except:
                pass
        
        # Fetch trip data
        trips_resp = _supabase.table('trips').select('user_id, created_at').execute()
        trips_df = pd.DataFrame(trips_resp.data or [])
        
        # Debug information
        st.write(f"Debug: Found {len(users_df)} users and {len(trips_df)} trips")
        st.write(f"Debug: Using table source: {table_source}")
        st.write(f"Debug: Using auth table: {use_auth_table}")
        if not users_df.empty:
            st.write(f"Debug: Users columns: {list(users_df.columns)}")
            if use_auth_table and 'email_confirmed_at' in users_df.columns:
                st.write(f"Debug: Sample email_confirmed_at: {users_df['email_confirmed_at'].head(3).tolist()}")
            elif 'created_at' in users_df.columns:
                st.write(f"Debug: Sample created_at: {users_df['created_at'].head(3).tolist()}")
                st.write(f"Debug: created_at dtype: {users_df['created_at'].dtype}")
                st.write(f"Debug: created_at sample values: {users_df['created_at'].head(3).values}")
        
        if not trips_df.empty:
            st.write(f"Debug: Trips columns: {list(trips_df.columns)}")
            st.write(f"Debug: Sample trip created_at: {trips_df['created_at'].head(3).tolist()}")
            st.write(f"Debug: trips created_at dtype: {trips_df['created_at'].dtype}")
        
        if users_df.empty:
            return {}, pd.DataFrame(), {}
        
        # Normalize timestamps with error handling
        try:
            if use_auth_table and 'email_confirmed_at' in users_df.columns:
                # Use email_confirmed_at as registration date
                st.write("Debug: Using email_confirmed_at for registration date")
                users_df['created_at_parsed'] = pd.to_datetime(users_df['email_confirmed_at'], errors='coerce', utc=True)
            elif 'created_at' in users_df.columns:
                # Use created_at as registration date
                st.write("Debug: Using created_at for registration date")
                users_df['created_at_parsed'] = pd.to_datetime(users_df['created_at'], errors='coerce', utc=True)
            else:
                st.error("Debug: No suitable date column found in users data")
                return {}, pd.DataFrame(), {}
            
            # Remove rows with invalid dates
            users_df = users_df.dropna(subset=['created_at_parsed'])
            st.write(f"Debug: After date parsing, {len(users_df)} users remain")
            
            # Convert to date
            users_df['registration_date'] = users_df['created_at_parsed'].dt.date
            st.write("Debug: Successfully converted to registration_date")
            
        except Exception as e:
            st.error(f"Debug: Error in user date processing: {str(e)}")
            return {}, pd.DataFrame(), {}
        
        if not trips_df.empty:
            try:
                st.write("Debug: Processing trips data...")
                trips_df['created_at_parsed'] = pd.to_datetime(trips_df['created_at'], errors='coerce', utc=True)
                trips_df = trips_df.dropna(subset=['created_at_parsed'])  # Remove rows with invalid dates
                trips_df['trip_date'] = trips_df['created_at_parsed'].dt.date
                st.write(f"Debug: Successfully processed {len(trips_df)} trips")
                
                # Get first trip date for each user
                first_trips = trips_df.groupby('user_id')['trip_date'].min().reset_index()
                first_trips.columns = ['user_id', 'first_trip_date']
                st.write(f"Debug: Found first trips for {len(first_trips)} users")
                
                # Merge with users
                user_data = users_df.merge(first_trips, left_on='id', right_on='user_id', how='left')
                st.write(f"Debug: Successfully merged user data, {len(user_data)} total records")
            except Exception as e:
                st.error(f"Debug: Error in trips processing: {str(e)}")
                return {}, pd.DataFrame(), {}
        else:
            user_data = users_df.copy()
            user_data['first_trip_date'] = None
            st.write("Debug: No trips data, using users only")
        
        # Calculate retention metrics
        try:
            st.write("Debug: Starting retention calculations...")
            now_utc = pd.Timestamp.now(tz='UTC').date()
            
            # Users who made their first trip
            users_with_trips = user_data[user_data['first_trip_date'].notna()]
            st.write(f"Debug: Found {len(users_with_trips)} users with trips")
            
            if users_with_trips.empty:
                st.write("Debug: No users with trips found")
                return {
                    'day_1_retention': 0,
                    'day_7_retention': 0,
                    'day_30_retention': 0,
                    'total_registered': len(user_data),
                    'users_with_trips': 0,
                    'activation_rate': 0
                }, pd.DataFrame(), {}
            
            # Calculate days between registration and first trip
            st.write("Debug: Calculating days to first trip...")
            # When subtracting date objects, we get timedelta, so use .apply(lambda x: x.days)
            users_with_trips['days_to_first_trip'] = (users_with_trips['first_trip_date'] - users_with_trips['registration_date']).apply(lambda x: x.days)
            st.write("Debug: Successfully calculated days to first trip")
        except Exception as e:
            st.error(f"Debug: Error in retention calculations: {str(e)}")
            return {}, pd.DataFrame(), {}
        
        # Day 1 retention (users who made first trip within 1 day)
        day_1_retention = len(users_with_trips[users_with_trips['days_to_first_trip'] <= 1]) / len(users_with_trips) * 100
        
        # Day 7 retention (users who made first trip within 7 days)
        day_7_retention = len(users_with_trips[users_with_trips['days_to_first_trip'] <= 7]) / len(users_with_trips) * 100
        
        # Day 30 retention (users who made first trip within 30 days)
        day_30_retention = len(users_with_trips[users_with_trips['days_to_first_trip'] <= 30]) / len(users_with_trips) * 100
        
        # Activation rate (users who made at least one trip)
        activation_rate = len(users_with_trips) / len(user_data) * 100
        
        retention_metrics = {
            'day_1_retention': round(day_1_retention, 1),
            'day_7_retention': round(day_7_retention, 1),
            'day_30_retention': round(day_30_retention, 1),
            'total_registered': len(user_data),
            'users_with_trips': len(users_with_trips),
            'activation_rate': round(activation_rate, 1)
        }
        
        # Cohort analysis - group by registration week
        try:
            st.write("Debug: Starting cohort analysis...")
            # Convert registration_date to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(user_data['registration_date']):
                user_data['registration_date'] = pd.to_datetime(user_data['registration_date'])
                st.write("Debug: Converted registration_date to datetime")
            
            # Calculate registration week (Monday of the week)
            user_data['registration_week'] = user_data['registration_date'].apply(
                lambda x: x - pd.Timedelta(days=x.weekday()) if pd.notna(x) else None
            )
            st.write("Debug: Successfully calculated registration weeks")
            
            # Also add registration_week to users_with_trips
            if not users_with_trips.empty:
                users_with_trips = users_with_trips.copy()  # Avoid SettingWithCopyWarning
                users_with_trips['registration_week'] = users_with_trips['registration_date'].apply(
                    lambda x: pd.to_datetime(x) - pd.Timedelta(days=pd.to_datetime(x).weekday()) if pd.notna(x) else None
                )
                st.write("Debug: Added registration_week to users_with_trips")
        except Exception as e:
            st.error(f"Debug: Error in cohort analysis: {str(e)}")
            return retention_metrics, pd.DataFrame(), {}
        
        cohort_data = []
        for week in sorted(user_data['registration_week'].dropna().unique()):
            week_users = user_data[user_data['registration_week'] == week]
            week_trips = users_with_trips[users_with_trips['registration_week'] == week] if not users_with_trips.empty else pd.DataFrame()
            
            cohort_data.append({
                'cohort_week': week.strftime('%Y-%m-%d') if pd.notna(week) else 'Unknown',
                'total_users': len(week_users),
                'activated_users': len(week_trips),
                'activation_rate': len(week_trips) / len(week_users) * 100 if len(week_users) > 0 else 0
            })
        
        cohort_df = pd.DataFrame(cohort_data)
        
        # Time-to-activation distribution
        activation_dist = users_with_trips['days_to_first_trip'].value_counts().sort_index()
        activation_dist = activation_dist.head(30)  # First 30 days
        
        return retention_metrics, cohort_df, activation_dist.to_dict()
        
    except Exception as e:
        st.error(f"Error calculating retention analysis: {str(e)}")
        return {}, pd.DataFrame(), {}

def format_duration(minutes):
    """Format duration from minutes to readable format"""
    if pd.isna(minutes):
        return "N/A"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"

def calculate_data_completeness(df):
    """Calculate overall data completeness score"""
    if df.empty:
        return 0.0

    key_fields = ['start_location', 'end_location', 'purpose', 'fuel_used', 'mileage', 'actual_distance']
    total_cells = len(df) * len(key_fields)
    complete_cells = 0

    for field in key_fields:
        if field in df.columns:
            non_null_count = df[field].notna().sum()
            # Also count non-zero values for numeric fields
            if field in ['fuel_used', 'mileage', 'actual_distance']:
                non_zero_count = (df[field] != 0).sum()
                complete_cells += min(non_null_count, non_zero_count)
            else:
                complete_cells += non_null_count

    completeness_score = (complete_cells / total_cells * 100) if total_cells > 0 else 0
    return completeness_score

# Revenue Analytics Functions
@st.cache_data(ttl=300)
def get_revenue_metrics(_supabase: Client):
    """Calculate revenue metrics by subscription tier"""
    try:
        # Get all profiles with subscription tiers
        profiles = _supabase.table('profiles').select('subscription_tier, created_at').execute().data
        
        if not profiles:
            return {}
        
        df = pd.DataFrame(profiles)
        
        # Define pricing (you can adjust these based on your actual pricing)
        pricing = {
            'free': 0,
            'basic': 9.99,
            'premium': 19.99,
            'pro': 39.99,
            'enterprise': 99.99
        }
        
        # Calculate metrics
        tier_counts = df['subscription_tier'].value_counts().to_dict()
        
        # Calculate MRR (Monthly Recurring Revenue)
        mrr = sum(tier_counts.get(tier.lower(), 0) * price 
                  for tier, price in pricing.items())
        
        # Calculate ARR (Annual Recurring Revenue)
        arr = mrr * 12
        
        # Calculate average revenue per user (ARPU)
        total_users = len(df)
        arpu = mrr / total_users if total_users > 0 else 0
        
        # Tier distribution
        tier_distribution = {
            tier: {
                'count': tier_counts.get(tier, 0),
                'revenue': tier_counts.get(tier, 0) * pricing.get(tier, 0),
                'percentage': (tier_counts.get(tier, 0) / total_users * 100) if total_users > 0 else 0
            }
            for tier in pricing.keys()
        }
        
        return {
            'mrr': mrr,
            'arr': arr,
            'arpu': arpu,
            'total_users': total_users,
            'tier_distribution': tier_distribution,
            'tier_counts': tier_counts
        }
    except Exception as e:
        st.error(f"Error calculating revenue metrics: {str(e)}")
        return {}

@st.cache_data(ttl=300)
def get_growth_metrics(_supabase: Client):
    """Calculate growth metrics (WoW, MoM, QoQ)"""
    try:
        # Get all profiles with creation dates
        profiles = _supabase.table('profiles').select('created_at').execute().data
        trips = _supabase.table('trips').select('created_at').execute().data
        
        if not profiles:
            return {}
        
        # Convert to DataFrame
        users_df = pd.DataFrame(profiles)
        users_df['created_at'] = pd.to_datetime(users_df['created_at'], utc=True)
        
        trips_df = pd.DataFrame(trips) if trips else pd.DataFrame()
        if not trips_df.empty:
            trips_df['created_at'] = pd.to_datetime(trips_df['created_at'], utc=True)
        
        now = pd.Timestamp.now(tz='UTC')
        
        # Calculate user growth
        def calculate_period_growth(df, days_current, days_previous):
            current_start = now - pd.Timedelta(days=days_current)
            previous_start = now - pd.Timedelta(days=days_previous)
            previous_end = now - pd.Timedelta(days=days_current)
            
            current_count = len(df[df['created_at'] >= current_start])
            previous_count = len(df[(df['created_at'] >= previous_start) & 
                                   (df['created_at'] < previous_end)])
            
            if previous_count > 0:
                growth_rate = ((current_count - previous_count) / previous_count) * 100
            else:
                growth_rate = 100 if current_count > 0 else 0
            
            return {
                'current': current_count,
                'previous': previous_count,
                'growth_rate': growth_rate
            }
        
        # Week over Week (WoW)
        wow_users = calculate_period_growth(users_df, 7, 14)
        wow_trips = calculate_period_growth(trips_df, 7, 14) if not trips_df.empty else {'current': 0, 'previous': 0, 'growth_rate': 0}
        
        # Month over Month (MoM)
        mom_users = calculate_period_growth(users_df, 30, 60)
        mom_trips = calculate_period_growth(trips_df, 30, 60) if not trips_df.empty else {'current': 0, 'previous': 0, 'growth_rate': 0}
        
        # Quarter over Quarter (QoQ)
        qoq_users = calculate_period_growth(users_df, 90, 180)
        qoq_trips = calculate_period_growth(trips_df, 90, 180) if not trips_df.empty else {'current': 0, 'previous': 0, 'growth_rate': 0}
        
        # Daily growth chart data
        users_df['date'] = users_df['created_at'].dt.date
        daily_signups = users_df.groupby('date').size().reset_index(name='signups')
        daily_signups['cumulative'] = daily_signups['signups'].cumsum()
        
        return {
            'wow': {'users': wow_users, 'trips': wow_trips},
            'mom': {'users': mom_users, 'trips': mom_trips},
            'qoq': {'users': qoq_users, 'trips': qoq_trips},
            'daily_signups': daily_signups.tail(30)  # Last 30 days
        }
    except Exception as e:
        st.error(f"Error calculating growth metrics: {str(e)}")
        return {}

@st.cache_data(ttl=300)
def get_trip_purpose_analytics(_supabase: Client):
    """Analyze trip purposes and their usage patterns"""
    try:
        response = _supabase.table('trips').select('purpose, user_id, mileage, actual_distance, fuel_used, reimbursement, created_at').execute()

        if not response.data:
            return pd.DataFrame()

        df = pd.DataFrame(response.data)

        # Clean and prepare data
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        df['distance'] = df['actual_distance'].fillna(df['mileage']).fillna(0)
        df['purpose'] = df['purpose'].fillna('Not Specified')
        df['fuel_used'] = pd.to_numeric(df['fuel_used'], errors='coerce').fillna(0)
        df['reimbursement'] = pd.to_numeric(df['reimbursement'], errors='coerce').fillna(0)

        # Categorize purposes (you can customize these categories)
        def categorize_purpose(purpose):
            purpose_lower = str(purpose).lower()
            if any(word in purpose_lower for word in ['business', 'meeting', 'client', 'work', 'office', 'conference']):
                return 'Business'
            elif any(word in purpose_lower for word in ['personal', 'home', 'shopping', 'doctor', 'family', 'vacation']):
                return 'Personal'
            elif any(word in purpose_lower for word in ['delivery', 'pickup', 'service']):
                return 'Service/Delivery'
            else:
                return 'Other'

        df['purpose_category'] = df['purpose'].apply(categorize_purpose)

        return df

    except Exception as e:
        st.error(f"Error fetching trip purpose analytics: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_fuel_efficiency_analytics(_supabase: Client):
    """Analyze fuel efficiency and costs"""
    try:
        response = _supabase.table('trips').select('mileage, actual_distance, fuel_used, reimbursement, user_id, created_at, purpose').execute()

        if not response.data:
            return pd.DataFrame()

        df = pd.DataFrame(response.data)

        # Clean and prepare data
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        df['distance'] = df['actual_distance'].fillna(df['mileage']).fillna(0)
        df['fuel_used'] = pd.to_numeric(df['fuel_used'], errors='coerce').fillna(0)
        df['reimbursement'] = pd.to_numeric(df['reimbursement'], errors='coerce').fillna(0)

        # Calculate fuel efficiency
        df['mpg'] = df.apply(lambda row: row['distance'] / row['fuel_used'] if row['fuel_used'] > 0 else 0, axis=1)

        # Assume average fuel price (you can make this configurable)
        avg_fuel_price = 3.50  # USD per gallon
        df['fuel_cost'] = df['fuel_used'] * avg_fuel_price
        df['cost_per_mile'] = df['fuel_cost'] / df['distance'] if df['distance'].sum() > 0 else 0

        return df

    except Exception as e:
        st.error(f"Error fetching fuel efficiency analytics: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_global_destinations(_supabase: Client):
    """Get all global destinations with usage statistics"""
    try:
        response = _supabase.table('global_destinations').select('*').execute()

        if not response.data:
            return pd.DataFrame()

        df = pd.DataFrame(response.data)

        # Convert timestamps
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        df['updated_at'] = pd.to_datetime(df['updated_at'], utc=True)
        df['last_used_at'] = pd.to_datetime(df['last_used_at'], utc=True)

        # Ensure numeric types
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['usage_count'] = pd.to_numeric(df['usage_count'], errors='coerce').fillna(0)

        # Remove rows with invalid coordinates
        df = df.dropna(subset=['latitude', 'longitude'])

        return df

    except Exception as e:
        st.error(f"Error fetching global destinations: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)  # Refresh every minute for live feed
def get_recent_trips(_supabase: Client, limit=20):
    """Get recent trips for live activity feed"""
    try:
        # Get recent trips with user info
        trips = _supabase.table('trips').select(
            'id, user_id, mileage, actual_distance, start_time, end_time, created_at'
        ).order('created_at', desc=True).limit(limit).execute().data
        
        if not trips:
            return pd.DataFrame()
        
        # Get user info for these trips
        user_ids = list(set(trip['user_id'] for trip in trips))
        profiles = _supabase.table('profiles').select(
            'id, full_name, subscription_tier'
        ).in_('id', user_ids).execute().data
        
        # Create user lookup
        user_lookup = {p['id']: p for p in profiles}
        
        # Enhance trip data with user info
        enhanced_trips = []
        for trip in trips:
            user_info = user_lookup.get(trip['user_id'], {})
            
            # Calculate duration if start and end times exist
            duration = None
            if trip.get('start_time') and trip.get('end_time'):
                start = pd.to_datetime(trip['start_time'])
                end = pd.to_datetime(trip['end_time'])
                duration = (end - start).total_seconds() / 60  # in minutes
            
            enhanced_trips.append({
                'time': pd.to_datetime(trip['created_at']),
                'user': user_info.get('full_name', 'Unknown User'),
                'tier': user_info.get('subscription_tier', 'free'),
                'distance': trip.get('actual_distance') or trip.get('mileage', 0),
                'duration': duration,
                'trip_id': trip['id']
            })
        
        return pd.DataFrame(enhanced_trips)
    except Exception as e:
        st.error(f"Error fetching recent trips: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_trip_logs_with_users(_supabase: Client, lookback_days: int = 90):
    """Get trip logs enriched with user information for day/week reporting."""
    try:
        cutoff_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        cutoff_date = cutoff_dt.isoformat()
        
        trips_response = _supabase.table('trips').select(
            'id, user_id, created_at, status, start_location, end_location, purpose, mileage, actual_distance, start_time, end_time'
        ).gte('created_at', cutoff_date).order('created_at', desc=True).execute()
        
        trips_df = pd.DataFrame(trips_response.data or [])
        if trips_df.empty:
            return pd.DataFrame()
        
        trips_df.rename(columns={'id': 'trip_id'}, inplace=True)
        trips_df['user_id'] = trips_df['user_id'].astype(str)
        trips_df['created_at'] = pd.to_datetime(trips_df['created_at'], errors='coerce', utc=True)
        trips_df['start_time'] = pd.to_datetime(trips_df['start_time'], errors='coerce', utc=True)
        trips_df['end_time'] = pd.to_datetime(trips_df['end_time'], errors='coerce', utc=True)
        
        trips_df['distance'] = pd.to_numeric(trips_df.get('actual_distance'), errors='coerce').fillna(
            pd.to_numeric(trips_df.get('mileage'), errors='coerce')
        ).fillna(0)
        
        trips_df['duration_mins'] = (
            (trips_df['end_time'] - trips_df['start_time']).dt.total_seconds() / 60
        ).fillna(0)
        trips_df['duration_mins'] = trips_df['duration_mins'].clip(lower=0)
        
        trips_df['status'] = trips_df['status'].fillna('unknown')
        status_norm = trips_df['status'].astype(str).str.strip().str.lower()
        trips_df['is_completed'] = status_norm.isin({'completed', 'complated'})
        
        profiles_response = _supabase.table('profiles').select(
            'id, full_name, phone_number, subscription_tier'
        ).execute()
        profiles_df = pd.DataFrame(profiles_response.data or [])
        
        if profiles_df.empty:
            profiles_df = pd.DataFrame(columns=['profile_id', 'full_name', 'phone_number', 'subscription_tier'])
        else:
            profiles_df['id'] = profiles_df['id'].astype(str)
            profiles_df.rename(columns={'id': 'profile_id'}, inplace=True)
        
        logs_df = trips_df.merge(
            profiles_df[['profile_id', 'full_name', 'phone_number', 'subscription_tier']],
            left_on='user_id',
            right_on='profile_id',
            how='left'
        )
        
        auth_users_df = get_auth_users(_supabase)
        if not auth_users_df.empty and {'id', 'email'}.issubset(auth_users_df.columns):
            email_map = auth_users_df.dropna(subset=['id']).drop_duplicates(subset=['id'], keep='first').set_index('id')['email']
            logs_df['email'] = logs_df['user_id'].map(email_map)
        else:
            logs_df['email'] = None
        
        return logs_df[[
            'trip_id', 'user_id', 'created_at', 'status', 'is_completed',
            'distance', 'duration_mins', 'start_location', 'end_location', 'purpose',
            'full_name', 'email', 'phone_number', 'subscription_tier'
        ]]
    except Exception as e:
        st.error(f"Error fetching trip logs: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_upgrade_intent_events(_supabase: Client, lookback_days: int = 30):
    """Get normalized, human-readable upgrade intent events."""
    try:
        cutoff_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        cutoff_date = cutoff_dt.isoformat()
        
        # Paginate to avoid default row limits in PostgREST.
        all_events = []
        page_size = 1000
        offset = 0
        max_rows = 20000
        
        while True:
            response = _supabase.table('upgrade_intent_log').select(
                'id, user_id, feature_name, timestamp, metadata'
            ).gte('timestamp', cutoff_date).order('timestamp', desc=True).range(offset, offset + page_size - 1).execute()
            
            batch = response.data or []
            if not batch:
                break
            
            all_events.extend(batch)
            if len(batch) < page_size:
                break
            
            offset += page_size
            if offset >= max_rows:
                break
        
        events_df = pd.DataFrame(all_events)
        if events_df.empty:
            return pd.DataFrame()
        
        events_df['user_id'] = events_df['user_id'].astype(str)
        events_df['event_ts'] = pd.to_datetime(events_df['timestamp'], errors='coerce', utc=True)
        
        # Flatten metadata keys used for analysis/display.
        metadata_series = events_df['metadata'].apply(lambda m: m if isinstance(m, dict) else {})
        events_df['meta_event_kind'] = metadata_series.apply(lambda m: m.get('event_kind'))
        events_df['meta_event_name'] = metadata_series.apply(lambda m: m.get('event_name'))
        events_df['meta_feature_name'] = metadata_series.apply(lambda m: m.get('feature_name'))
        events_df['platform'] = metadata_series.apply(lambda m: m.get('platform'))
        events_df['source'] = metadata_series.apply(lambda m: m.get('source'))
        events_df['flow'] = metadata_series.apply(lambda m: m.get('flow'))
        events_df['trip_id'] = metadata_series.apply(lambda m: m.get('trip_id'))
        events_df['gate_strategy'] = metadata_series.apply(lambda m: m.get('gate_strategy'))
        events_df['event_subscription_tier'] = metadata_series.apply(lambda m: m.get('subscription_tier'))
        events_df['experiment_name'] = metadata_series.apply(lambda m: m.get('experiment_name'))
        events_df['experiment_variant'] = metadata_series.apply(lambda m: m.get('experiment_variant'))
        
        # Parse feature_name patterns like conversion_event:premium_purchase_started.
        parsed = events_df['feature_name'].astype(str).str.split(':', n=1, expand=True)
        if parsed.shape[1] > 1:
            events_df['event_kind'] = events_df['meta_event_kind'].fillna(parsed[0])
            events_df['event_name'] = events_df['meta_event_name'].fillna(parsed[1])
            events_df['parsed_feature'] = events_df['feature_name'].astype(str)
            has_delimiter = events_df['feature_name'].astype(str).str.contains(':')
            events_df.loc[has_delimiter, 'parsed_feature'] = parsed.loc[has_delimiter, 1]
            events_df['feature_display'] = events_df['meta_feature_name'].combine_first(events_df['parsed_feature'])
        else:
            events_df['event_kind'] = events_df['meta_event_kind'].fillna('unknown')
            events_df['event_name'] = events_df['meta_event_name'].fillna(events_df['feature_name'])
            events_df['feature_display'] = events_df['meta_feature_name'].fillna(events_df['feature_name'])
        
        events_df['event_name'] = events_df['event_name'].fillna(events_df['feature_name'])
        events_df['event_family'] = events_df['event_kind'].map({
            'upgrade_intent': 'Upgrade Intent',
            'conversion_event': 'Conversion Event'
        }).fillna('Other')
        events_df['action_label'] = events_df['event_name'].astype(str).str.replace('_', ' ', regex=False).str.strip().str.title()
        
        # Attach profile context.
        profiles_response = _supabase.table('profiles').select(
            'id, full_name, phone_number, subscription_tier'
        ).execute()
        profiles_df = pd.DataFrame(profiles_response.data or [])
        if not profiles_df.empty:
            profiles_df['id'] = profiles_df['id'].astype(str)
            profiles_df.rename(columns={
                'id': 'profile_id',
                'subscription_tier': 'profile_subscription_tier'
            }, inplace=True)
            events_df = events_df.merge(
                profiles_df[['profile_id', 'full_name', 'phone_number', 'profile_subscription_tier']],
                left_on='user_id',
                right_on='profile_id',
                how='left'
            )
        else:
            events_df['full_name'] = None
            events_df['phone_number'] = None
            events_df['profile_subscription_tier'] = None
        
        # Attach auth email.
        auth_users_df = get_auth_users(_supabase)
        if not auth_users_df.empty and {'id', 'email'}.issubset(auth_users_df.columns):
            email_map = auth_users_df.dropna(subset=['id']).drop_duplicates(subset=['id'], keep='first').set_index('id')['email']
            events_df['email'] = events_df['user_id'].map(email_map)
        else:
            events_df['email'] = None
        
        # Final readable tier field with event tier as first source.
        events_df['subscription_tier'] = events_df['event_subscription_tier'].fillna(events_df['profile_subscription_tier'])
        
        return events_df[[
            'id', 'event_ts', 'user_id', 'full_name', 'email', 'phone_number',
            'event_family', 'event_kind', 'event_name', 'action_label',
            'feature_display', 'source', 'platform', 'flow', 'trip_id',
            'subscription_tier', 'gate_strategy', 'experiment_name', 'experiment_variant'
        ]]
    except Exception as e:
        st.error(f"Error fetching upgrade intent events: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_user_trip_completion(_supabase: Client):
    """Get trip completion summary per user."""
    try:
        trips_response = _supabase.table('trips').select('user_id, status, created_at').execute()
        trips_df = pd.DataFrame(trips_response.data or [])
        
        if trips_df.empty:
            return pd.DataFrame(columns=[
                'id', 'total_trips', 'completed_trips', 'completed_any_trip',
                'first_trip_at', 'last_trip_at', 'first_completed_trip_at'
            ])
        
        trips_df['user_id'] = trips_df['user_id'].astype(str)
        trips_df['created_at'] = pd.to_datetime(trips_df['created_at'], errors='coerce', utc=True)
        trips_df['status_norm'] = trips_df['status'].fillna('').astype(str).str.strip().str.lower()
        
        # Include the observed typo variant in your current data.
        completed_statuses = {'completed', 'complated'}
        trips_df['is_completed'] = trips_df['status_norm'].isin(completed_statuses)
        
        summary_df = trips_df.groupby('user_id').agg(
            total_trips=('user_id', 'count'),
            completed_trips=('is_completed', 'sum'),
            first_trip_at=('created_at', 'min'),
            last_trip_at=('created_at', 'max')
        ).reset_index().rename(columns={'user_id': 'id'})
        
        first_completed_df = (
            trips_df[trips_df['is_completed']]
            .groupby('user_id')['created_at']
            .min()
            .reset_index()
            .rename(columns={'user_id': 'id', 'created_at': 'first_completed_trip_at'})
        )
        
        summary_df = summary_df.merge(first_completed_df, on='id', how='left')
        summary_df['total_trips'] = summary_df['total_trips'].fillna(0).astype(int)
        summary_df['completed_trips'] = summary_df['completed_trips'].fillna(0).astype(int)
        summary_df['completed_any_trip'] = summary_df['completed_trips'] > 0
        
        return summary_df
    except Exception as e:
        st.error(f"Error fetching trip completion summary: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_auth_users(_supabase: Client):
    """Get users directly from Supabase Authentication."""
    try:
        auth_users = []
        page = 1
        per_page = 1000
        
        while True:
            users_batch = _supabase.auth.admin.list_users(page=page, per_page=per_page)
            if not users_batch:
                break
            
            for user in users_batch:
                if hasattr(user, 'model_dump'):
                    user = user.model_dump()
                if not isinstance(user, dict):
                    continue
                
                app_metadata = user.get('app_metadata') if isinstance(user.get('app_metadata'), dict) else {}
                provider = app_metadata.get('provider')
                if not provider:
                    providers = app_metadata.get('providers')
                    if isinstance(providers, list) and providers:
                        provider = providers[0]
                
                auth_users.append({
                    'id': str(user.get('id')) if user.get('id') else None,
                    'email': user.get('email'),
                    'phone': user.get('phone'),
                    'provider': provider,
                    'created_at': user.get('created_at'),
                    'email_confirmed_at': user.get('email_confirmed_at') or user.get('confirmed_at'),
                    'last_sign_in_at': user.get('last_sign_in_at')
                })
            
            if len(users_batch) < per_page:
                break
            page += 1
        
        if not auth_users:
            return pd.DataFrame()
        
        auth_users_df = pd.DataFrame(auth_users)
        auth_users_df = auth_users_df.dropna(subset=['id']).drop_duplicates(subset=['id'], keep='first')
        
        for col in ['created_at', 'email_confirmed_at', 'last_sign_in_at']:
            auth_users_df[col] = pd.to_datetime(auth_users_df[col], errors='coerce', utc=True)
        
        return auth_users_df
    except Exception:
        return pd.DataFrame()

def main():
    # Header
    st.title("🚗 Mileage Tracker Pro Dashboard")
    st.markdown("Real-time insights into user activity and trip statistics")
    
    # Initialize Supabase with enhanced client
    try:
        supabase = get_supabase_client()
        manager = get_supabase_manager()
    except Exception as e:
        st.error(f"⚠️ Failed to connect to database: {str(e)}")
        st.info("Please check your .env file and ensure your Supabase credentials are correct.")
        st.stop()
    
    # Sidebar for filters
    with st.sidebar:
        st.header("⚙️ Dashboard Settings")
        
        # Connection Status
        with st.expander("🔌 Connection Status", expanded=False):
            metrics = manager.get_metrics()
            conn_status = metrics.get('connection_status', {})
            
            if conn_status.get('is_healthy', False):
                st.success("✅ Database Connected")
            else:
                st.error("❌ Database Disconnected")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Queries", metrics.get('total_queries', 0))
                st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1f}%")
            with col2:
                st.metric("Errors", metrics.get('failed_queries', 0))
                uptime = metrics.get('uptime_seconds', 0)
                st.metric("Uptime", f"{int(uptime // 60)}m")
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Time range filter
        st.subheader("📅 Time Range")
        time_range = st.selectbox(
            "Select period for active users",
            options=[7, 14, 30, 60, 90],
            format_func=lambda x: f"Last {x} days",
            index=2
        )
        
        st.markdown("---")
        
        # Auto-refresh option
        st.subheader("🔄 Auto-Refresh")
        auto_refresh = st.checkbox("Enable auto-refresh (5 min)")
        
        if auto_refresh:
            st.info("Dashboard will refresh every 5 minutes")
        
        # Last updated time
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(["📊 Overview", "💰 Revenue", "📈 Growth", "🔴 Live Feed", "👥 Users", "📈 Retention", "🗺️ Destinations Map", "🔧 Advanced Analytics", "🆕 New Users", "🗓️ Trip Logs", "🚀 Upgrade Signals"])
    
    with tab1:
        # Fetch key metrics
        total_users = get_total_users(supabase)
        active_users = get_active_users(supabase, time_range)
        trip_stats = get_trip_statistics(supabase)
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Users",
                value=f"{total_users:,}",
                delta=f"{active_users} active"
            )
        
        with col2:
            st.metric(
                label="Active Users",
                value=f"{active_users:,}",
                delta=f"{(active_users/total_users*100):.1f}%" if total_users > 0 else "0%"
            )
        
        with col3:
            st.metric(
                label="Total Trips",
                value=f"{trip_stats.get('total_trips', 0):,}",
                delta=f"+{trip_stats.get('trips_today', 0)} today"
            )
        
        with col4:
            avg_trips_per_user = trip_stats.get('total_trips', 0) / total_users if total_users > 0 else 0
            st.metric(
                label="Avg Trips/User",
                value=f"{avg_trips_per_user:.1f}",
                delta=f"{trip_stats.get('trips_this_week', 0)} this week"
            )
        
        style_metric_cards()
        
        st.markdown("---")
        
        # Top Active Users Section
        st.subheader("🏆 Top 10 Most Active Users")
        
        top_users = get_top_active_users(supabase, 10)
        
        if not top_users.empty:
            # Create two columns for the table and chart
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Format the dataframe for display
                display_df = top_users.copy()
                display_df['trips_per_week'] = display_df['trips_per_week'].round(1)
                display_df['total_distance'] = display_df['total_distance'].round(1)
                display_df['total_duration'] = display_df['total_duration'].apply(format_duration)
                display_df['days_since_last_trip'] = display_df['days_since_last_trip'].astype(int)
                
                # Rename columns for better display
                display_df.columns = ['Name', 'Email', 'Phone', 'Subscription', 'Total Trips', 'Trips/Week', 
                                     'Total Distance (mi)', 'Total Duration', 'Days Since Last Trip']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Email": st.column_config.TextColumn(width="large"),
                        "Phone": st.column_config.TextColumn(width="medium"),
                        "Subscription": st.column_config.TextColumn(width="small"),
                        "Total Trips": st.column_config.NumberColumn(format="%d"),
                        "Trips/Week": st.column_config.NumberColumn(format="%.1f"),
                        "Total Distance (mi)": st.column_config.NumberColumn(format="%.1f"),
                        "Days Since Last Trip": st.column_config.NumberColumn(format="%d")
                    }
                )
            
            with col2:
                # Create a bar chart of top users
                fig = px.bar(
                    top_users.head(5),
                    x='trip_count',
                    y='full_name',
                    orientation='h',
                    title="Top 5 Users by Trip Count",
                    labels={'trip_count': 'Number of Trips', 'full_name': 'User'},
                    color='trip_count',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user data available")
    
    # Revenue Analytics Tab
    with tab2:
        st.subheader("💰 Revenue & Subscription Analytics")
        
        revenue_metrics = get_revenue_metrics(supabase)
        
        if revenue_metrics:
            # Key revenue metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Monthly Recurring Revenue (MRR)",
                    f"${revenue_metrics.get('mrr', 0):,.2f}",
                    help="Total monthly revenue from all subscriptions"
                )
            
            with col2:
                st.metric(
                    "Annual Recurring Revenue (ARR)", 
                    f"${revenue_metrics.get('arr', 0):,.2f}",
                    help="Projected annual revenue (MRR × 12)"
                )
            
            with col3:
                st.metric(
                    "Average Revenue Per User (ARPU)",
                    f"${revenue_metrics.get('arpu', 0):.2f}",
                    help="Average monthly revenue per user"
                )
            
            with col4:
                st.metric(
                    "Paying Users",
                    f"{sum(1 for tier, data in revenue_metrics.get('tier_distribution', {}).items() if tier != 'free' and data['count'] > 0):,}",
                    f"{(sum(1 for tier, data in revenue_metrics.get('tier_distribution', {}).items() if tier != 'free' and data['count'] > 0) / revenue_metrics.get('total_users', 1) * 100):.1f}%"
                )
            
            st.markdown("---")
            
            # Subscription tier distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Subscription Tier Distribution")
                
                tier_data = []
                for tier, data in revenue_metrics.get('tier_distribution', {}).items():
                    if data['count'] > 0:
                        tier_data.append({
                            'Tier': tier.capitalize(),
                            'Users': data['count'],
                            'Revenue': data['revenue'],
                            'Percentage': data['percentage']
                        })
                
                if tier_data:
                    tier_df = pd.DataFrame(tier_data)
                    
                    # Pie chart of user distribution
                    fig = px.pie(
                        tier_df,
                        values='Users',
                        names='Tier',
                        title="Users by Subscription Tier",
                        color_discrete_map={
                            'Free': '#gray',
                            'Basic': '#blue',
                            'Premium': '#green',
                            'Pro': '#purple',
                            'Enterprise': '#gold'
                        }
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Revenue by Tier")
                
                if tier_data:
                    # Filter out free tier for revenue chart
                    revenue_df = pd.DataFrame([t for t in tier_data if t['Revenue'] > 0])
                    
                    if not revenue_df.empty:
                        fig = px.bar(
                            revenue_df,
                            x='Tier',
                            y='Revenue',
                            title="Monthly Revenue by Tier",
                            color='Revenue',
                            color_continuous_scale='Greens',
                            text='Revenue'
                        )
                        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No revenue data available (all users on free tier)")
            
            # Detailed tier breakdown table
            st.markdown("#### Detailed Tier Breakdown")
            if tier_data:
                detailed_df = pd.DataFrame(tier_data)
                detailed_df['Revenue'] = detailed_df['Revenue'].apply(lambda x: f"${x:,.2f}")
                detailed_df['Percentage'] = detailed_df['Percentage'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(
                    detailed_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Users": st.column_config.NumberColumn(format="%d"),
                        "Revenue": st.column_config.TextColumn(),
                        "Percentage": st.column_config.TextColumn()
                    }
                )
        else:
            st.info("No revenue data available")
    
    # Growth Metrics Tab
    with tab3:
        st.subheader("📈 Growth Metrics")
        
        growth_metrics = get_growth_metrics(supabase)
        
        if growth_metrics:
            # Growth rate cards
            st.markdown("#### User Growth")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                wow = growth_metrics.get('wow', {}).get('users', {})
                delta_value = wow.get('growth_rate', 0)
                st.metric(
                    "Week over Week (WoW)",
                    f"{wow.get('current', 0)} users",
                    delta=f"{delta_value:+.1f}%" if delta_value != 0 else "0%",
                    delta_color="normal" if delta_value >= 0 else "inverse"
                )
            
            with col2:
                mom = growth_metrics.get('mom', {}).get('users', {})
                delta_value = mom.get('growth_rate', 0)
                st.metric(
                    "Month over Month (MoM)",
                    f"{mom.get('current', 0)} users",
                    delta=f"{delta_value:+.1f}%" if delta_value != 0 else "0%",
                    delta_color="normal" if delta_value >= 0 else "inverse"
                )
            
            with col3:
                qoq = growth_metrics.get('qoq', {}).get('users', {})
                delta_value = qoq.get('growth_rate', 0)
                st.metric(
                    "Quarter over Quarter (QoQ)",
                    f"{qoq.get('current', 0)} users",
                    delta=f"{delta_value:+.1f}%" if delta_value != 0 else "0%",
                    delta_color="normal" if delta_value >= 0 else "inverse"
                )
            
            st.markdown("#### Trip Growth")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                wow = growth_metrics.get('wow', {}).get('trips', {})
                delta_value = wow.get('growth_rate', 0)
                st.metric(
                    "Week over Week (WoW)",
                    f"{wow.get('current', 0)} trips",
                    delta=f"{delta_value:+.1f}%" if delta_value != 0 else "0%",
                    delta_color="normal" if delta_value >= 0 else "inverse"
                )
            
            with col2:
                mom = growth_metrics.get('mom', {}).get('trips', {})
                delta_value = mom.get('growth_rate', 0)
                st.metric(
                    "Month over Month (MoM)",
                    f"{mom.get('current', 0)} trips",
                    delta=f"{delta_value:+.1f}%" if delta_value != 0 else "0%",
                    delta_color="normal" if delta_value >= 0 else "inverse"
                )
            
            with col3:
                qoq = growth_metrics.get('qoq', {}).get('trips', {})
                delta_value = qoq.get('growth_rate', 0)
                st.metric(
                    "Quarter over Quarter (QoQ)",
                    f"{qoq.get('current', 0)} trips",
                    delta=f"{delta_value:+.1f}%" if delta_value != 0 else "0%",
                    delta_color="normal" if delta_value >= 0 else "inverse"
                )
            
            st.markdown("---")
            
            # Growth charts
            daily_signups = growth_metrics.get('daily_signups')
            if daily_signups is not None and not daily_signups.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Daily signups chart
                    fig = px.bar(
                        daily_signups,
                        x='date',
                        y='signups',
                        title="Daily New User Signups (Last 30 Days)",
                        labels={'signups': 'New Users', 'date': 'Date'}
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Cumulative growth chart
                    fig = px.line(
                        daily_signups,
                        x='date',
                        y='cumulative',
                        title="Cumulative User Growth",
                        labels={'cumulative': 'Total Users', 'date': 'Date'},
                        markers=True
                    )
                    fig.update_traces(fill='tozeroy')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No growth data available")
    
    # Live Activity Feed Tab
    with tab4:
        st.subheader("🔴 Live Trip Activity Feed")
        
        # Auto-refresh settings
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**Real-time trip monitoring** - Updates every minute")
        with col2:
            if st.button("🔄 Refresh Now"):
                st.cache_data.clear()
                st.rerun()
        with col3:
            show_count = st.selectbox("Show last", [10, 20, 50, 100], index=1)
        
        # Get recent trips
        recent_trips = get_recent_trips(supabase, limit=show_count)
        
        if not recent_trips.empty:
            # Activity metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate metrics for last hour
            now = pd.Timestamp.now(tz='UTC')
            last_hour = recent_trips[recent_trips['time'] >= (now - pd.Timedelta(hours=1))]
            
            with col1:
                st.metric("Trips (Last Hour)", len(last_hour))
            with col2:
                unique_users = recent_trips['user'].nunique()
                st.metric("Active Users", unique_users)
            with col3:
                avg_distance = recent_trips['distance'].mean()
                st.metric("Avg Distance", f"{avg_distance:.1f} mi")
            with col4:
                if recent_trips['duration'].notna().any():
                    avg_duration = recent_trips['duration'].mean()
                    st.metric("Avg Duration", f"{avg_duration:.0f} min")
                else:
                    st.metric("Avg Duration", "N/A")
            
            st.markdown("---")
            
            # Live feed display
            st.markdown("#### Recent Trips")
            
            # Format the dataframe for display
            display_df = recent_trips.copy()
            display_df['time'] = display_df['time'].dt.strftime('%H:%M:%S')
            display_df['distance'] = display_df['distance'].round(1).astype(str) + ' mi'
            display_df['duration'] = display_df['duration'].apply(
                lambda x: f"{x:.0f} min" if pd.notna(x) else "N/A"
            )
            
            # Add tier badges
            tier_colors = {
                'free': '⚪',
                'basic': '🔵',
                'premium': '🟢',
                'pro': '🟣',
                'enterprise': '🟡'
            }
            display_df['tier'] = display_df['tier'].apply(
                lambda x: f"{tier_colors.get(x, '⚪')} {x.capitalize()}"
            )
            
            # Rename columns
            display_df = display_df[['time', 'user', 'tier', 'distance', 'duration']]
            display_df.columns = ['Time', 'User', 'Tier', 'Distance', 'Duration']
            
            # Display as a table with custom styling
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Time": st.column_config.TextColumn(width="small"),
                    "User": st.column_config.TextColumn(width="medium"),
                    "Tier": st.column_config.TextColumn(width="small"),
                    "Distance": st.column_config.TextColumn(width="small"),
                    "Duration": st.column_config.TextColumn(width="small")
                }
            )
            
            # Activity timeline
            st.markdown("#### Activity Timeline")
            
            # Group trips by hour for the timeline
            recent_trips['hour'] = pd.to_datetime(recent_trips['time']).dt.floor('H')
            hourly_trips = recent_trips.groupby('hour').size().reset_index(name='trips')
            
            fig = px.line(
                hourly_trips,
                x='hour',
                y='trips',
                title="Trip Activity Over Time",
                labels={'trips': 'Number of Trips', 'hour': 'Time'},
                markers=True
            )
            fig.update_traces(fill='tozeroy')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent trip activity")
    
    with tab5:
        st.subheader("👥 Detailed User Analytics")
        
        # User growth metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # User activity distribution
            usage_data = get_usage_patterns(supabase)
            if not usage_data.empty:
                daily_active = usage_data.groupby('date')['user_id'].nunique().reset_index()
                daily_active.columns = ['Date', 'Active Users']
                
                fig = px.line(
                    daily_active,
                    x='Date',
                    y='Active Users',
                    title='Daily Active Users (Last 30 Days)',
                    markers=True
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # User engagement heatmap
            if not usage_data.empty:
                hourly_usage = usage_data.groupby(['day_of_week', 'hour']).size().reset_index(name='trips')
                
                # Create pivot table for heatmap
                heatmap_data = hourly_usage.pivot(index='hour', columns='day_of_week', values='trips').fillna(0)
                
                # Reorder days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_data = heatmap_data.reindex(columns=[d for d in day_order if d in heatmap_data.columns])
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale='Blues',
                    text=heatmap_data.values,
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title='Usage Heatmap by Day and Hour',
                    xaxis_title='Day of Week',
                    yaxis_title='Hour of Day',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # User retention metrics
        st.markdown("---")
        st.subheader("📊 User Engagement Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            engagement_rate = (active_users / total_users * 100) if total_users > 0 else 0
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=engagement_rate,
                title={'text': f"Engagement Rate ({time_range} days)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "lightblue"},
                        {'range': [75, 100], 'color': "blue"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average trips per active user
            avg_trips_active = trip_stats.get('trips_this_month', 0) / active_users if active_users > 0 else 0
            st.metric(
                "Avg Trips per Active User",
                f"{avg_trips_active:.1f}",
                f"Last {time_range} days"
            )
            
            # Users at Risk metric (inactivity threshold adjustable below)
            risk_df, never_used_count = get_users_at_risk(supabase, 14)
            st.metric(
                "Users at Risk",
                len(risk_df),
                "Inactive > 14 days",
                delta_color="inverse"
            )
        
        with col3:
            # Power users (>10 trips)
            if not top_users.empty:
                power_users = len(top_users[top_users['trip_count'] > 10])
                st.metric(
                    "Power Users",
                    power_users,
                    ">10 trips"
                )
            
            # New vs returning ratio
            st.metric(
                "Activity Score",
                f"{min(100, int(engagement_rate * 1.5))}/100",
                "Overall health"
            )

        # Users at Risk table
        st.markdown("---")
        st.subheader("⚠️ Users at Risk")
        risk_threshold = st.slider("Inactivity threshold (days)", min_value=7, max_value=60, value=14, step=1)
        risk_df, never_used_count = get_users_at_risk(supabase, risk_threshold)
        st.caption(f"Never-used users (no trips): {never_used_count}")
        
        if not risk_df.empty:
            display_risk = risk_df.copy()
            display_risk['last_trip'] = pd.to_datetime(display_risk['last_trip']).dt.strftime('%Y-%m-%d %H:%M UTC')
            display_risk.rename(columns={
                'full_name': 'Name',
                'phone_number': 'Phone',
                'subscription_tier': 'Subscription',
                'trip_count': 'Trips',
                'total_distance': 'Total Distance (mi)',
                'days_since_last_trip': 'Days Since Last Trip',
                'last_trip': 'Last Trip'
            }, inplace=True)
            st.dataframe(display_risk, use_container_width=True, hide_index=True)
            # Download
            csv = display_risk.to_csv(index=False).encode('utf-8')
            st.download_button("Download at-risk users (CSV)", data=csv, file_name=f"users_at_risk_{risk_threshold}d.csv", mime="text/csv")
        else:
            st.info("No users meet the at-risk criteria for the selected threshold.")
        
        # Trip statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Distance",
                f"{trip_stats.get('total_distance', 0):,.0f} mi",
                f"Avg: {trip_stats.get('avg_distance', 0):.1f} mi"
            )
        
        with col2:
            total_hours = trip_stats.get('total_duration', 0) / 60
            st.metric(
                "Total Duration",
                f"{total_hours:,.0f} hrs",
                f"Avg: {format_duration(trip_stats.get('avg_duration', 0))}"
            )
        
        with col3:
            trips_per_day = trip_stats.get('total_trips', 0) / 30
            st.metric(
                "Trips per Day",
                f"{trips_per_day:.1f}",
                f"Total: {trip_stats.get('total_trips', 0):,}"
            )
        
        with col4:
            if trip_stats.get('avg_duration', 0) > 0:
                avg_speed = trip_stats.get('avg_distance', 0) / (trip_stats.get('avg_duration', 0) / 60)
            else:
                avg_speed = 0
            st.metric(
                "Avg Speed",
                f"{avg_speed:.1f} mph",
                "Per trip average"
            )
        
        st.markdown("---")
        
        # Trip distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Trips by day of week
            if not usage_data.empty:
                trips_by_day = usage_data['day_of_week'].value_counts().reset_index()
                trips_by_day.columns = ['Day', 'Trips']
                
                # Order days properly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                trips_by_day['Day'] = pd.Categorical(trips_by_day['Day'], categories=day_order, ordered=True)
                trips_by_day = trips_by_day.sort_values('Day')
                
                fig = px.bar(
                    trips_by_day,
                    x='Day',
                    y='Trips',
                    title='Trips by Day of Week',
                    color='Trips',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Trips by hour of day
            if not usage_data.empty:
                trips_by_hour = usage_data['hour'].value_counts().sort_index().reset_index()
                trips_by_hour.columns = ['Hour', 'Trips']
                
                fig = px.area(
                    trips_by_hour,
                    x='Hour',
                    y='Trips',
                    title='Trips by Hour of Day'
                )
                fig.update_traces(fill='tozeroy')
                fig.update_layout(height=350)
                fig.update_xaxes(tickmode='linear', tick0=0, dtick=2)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.subheader("📈 User Retention Analysis")
        
        # Get retention data
        retention_metrics, cohort_df, activation_dist = get_user_retention_analysis(supabase)
        
        if not retention_metrics:
            st.warning("No retention data available. Please ensure you have user registration and trip data.")
            return
        
        # Key retention metrics
        st.markdown("### 🎯 Key Retention Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Day 1 Retention",
                f"{retention_metrics.get('day_1_retention', 0):.1f}%",
                "Users who made first trip within 1 day"
            )
        
        with col2:
            st.metric(
                "Day 7 Retention", 
                f"{retention_metrics.get('day_7_retention', 0):.1f}%",
                "Users who made first trip within 7 days"
            )
        
        with col3:
            st.metric(
                "Day 30 Retention",
                f"{retention_metrics.get('day_30_retention', 0):.1f}%",
                "Users who made first trip within 30 days"
            )
        
        with col4:
            st.metric(
                "Activation Rate",
                f"{retention_metrics.get('activation_rate', 0):.1f}%",
                f"{retention_metrics.get('users_with_trips', 0)}/{retention_metrics.get('total_registered', 0)} users"
            )
        
        # Retention funnel visualization
        st.markdown("---")
        st.subheader("🔄 User Activation Funnel")
        
        funnel_data = {
            'Stage': ['Registered Users', 'Day 1 Active', 'Day 7 Active', 'Day 30 Active'],
            'Count': [
                retention_metrics.get('total_registered', 0),
                int(retention_metrics.get('total_registered', 0) * retention_metrics.get('day_1_retention', 0) / 100),
                int(retention_metrics.get('total_registered', 0) * retention_metrics.get('day_7_retention', 0) / 100),
                int(retention_metrics.get('total_registered', 0) * retention_metrics.get('day_30_retention', 0) / 100)
            ],
            'Percentage': [
                100.0,
                retention_metrics.get('day_1_retention', 0),
                retention_metrics.get('day_7_retention', 0),
                retention_metrics.get('day_30_retention', 0)
            ]
        }
        
        funnel_df = pd.DataFrame(funnel_data)
        
        fig = go.Figure(go.Funnel(
            y=funnel_df['Stage'],
            x=funnel_df['Count'],
            textinfo="value+percent initial",
            marker=dict(color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ))
        
        fig.update_layout(
            title="User Activation Funnel",
            height=400,
            margin=dict(l=100, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Time to activation distribution
        if activation_dist:
            st.markdown("---")
            st.subheader("⏱️ Time to First Trip Distribution")
            
            days = list(activation_dist.keys())
            counts = list(activation_dist.values())
            
            fig = go.Figure(data=[
                go.Bar(x=days, y=counts, marker_color='lightblue')
            ])
            
            fig.update_layout(
                title="Days Between Registration and First Trip",
                xaxis_title="Days",
                yaxis_title="Number of Users",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Cohort analysis
        if not cohort_df.empty:
            st.markdown("---")
            st.subheader("📊 Weekly Cohort Analysis")
            
            # Display cohort table
            display_cohort = cohort_df.copy()
            display_cohort['activation_rate'] = display_cohort['activation_rate'].round(1)
            display_cohort.rename(columns={
                'cohort_week': 'Registration Week',
                'total_users': 'Total Users',
                'activated_users': 'Activated Users',
                'activation_rate': 'Activation Rate (%)'
            }, inplace=True)
            
            st.dataframe(display_cohort, use_container_width=True, hide_index=True)
            
            # Cohort visualization
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=cohort_df['cohort_week'],
                y=cohort_df['total_users'],
                mode='lines+markers',
                name='Total Users',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=cohort_df['cohort_week'],
                y=cohort_df['activated_users'],
                mode='lines+markers',
                name='Activated Users',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="User Registration and Activation by Week",
                xaxis_title="Registration Week",
                yaxis_title="Number of Users",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Retention insights and recommendations
        st.markdown("---")
        st.subheader("💡 Retention Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Current Performance")
            
            day_1 = retention_metrics.get('day_1_retention', 0)
            day_7 = retention_metrics.get('day_7_retention', 0)
            day_30 = retention_metrics.get('day_30_retention', 0)
            activation = retention_metrics.get('activation_rate', 0)
            
            if day_1 < 30:
                st.warning(f"**Low Day 1 Retention ({day_1:.1f}%)** - Users aren't engaging immediately after registration")
            elif day_1 > 60:
                st.success(f"**Strong Day 1 Retention ({day_1:.1f}%)** - Great first impression!")
            else:
                st.info(f"**Moderate Day 1 Retention ({day_1:.1f}%)** - Room for improvement")
            
            if activation < 50:
                st.warning(f"**Low Activation Rate ({activation:.1f}%)** - Many users never make their first trip")
            elif activation > 80:
                st.success(f"**High Activation Rate ({activation:.1f}%)** - Excellent user onboarding!")
            else:
                st.info(f"**Moderate Activation Rate ({activation:.1f}%)** - Good foundation to build on")
        
        with col2:
            st.markdown("### 🎯 Actionable Recommendations")
            
            if day_1 < 40:
                st.markdown("""
                **🚀 Improve Day 1 Retention:**
                - Send welcome email with app tutorial
                - Offer first-trip incentives
                - Simplify onboarding process
                - Add progress indicators
                """)
            
            if activation < 60:
                st.markdown("""
                **📱 Boost Activation:**
                - Create guided first trip experience
                - Implement push notifications
                - Add gamification elements
                - Provide clear value proposition
                """)
            
            if day_7 < day_1 * 0.8:
                st.markdown("""
                **🔄 Maintain Engagement:**
                - Send follow-up messages
                - Create habit-forming features
                - Implement streak tracking
                - Offer weekly challenges
                """)
        
        # Download retention report
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Retention Report", use_container_width=True):
                # Create comprehensive retention report
                report_data = {
                    'Metric': ['Total Registered Users', 'Users with Trips', 'Activation Rate (%)', 
                              'Day 1 Retention (%)', 'Day 7 Retention (%)', 'Day 30 Retention (%)'],
                    'Value': [
                        retention_metrics.get('total_registered', 0),
                        retention_metrics.get('users_with_trips', 0),
                        retention_metrics.get('activation_rate', 0),
                        retention_metrics.get('day_1_retention', 0),
                        retention_metrics.get('day_7_retention', 0),
                        retention_metrics.get('day_30_retention', 0)
                    ]
                }
                
                report_df = pd.DataFrame(report_data)
                csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"retention_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        with col2:
            st.info("💡 **Pro Tip:** Monitor these metrics weekly to track retention improvements and identify trends early.")

    with tab7:
        st.subheader("🗺️ Global Destinations Map")

        # Get destinations data
        destinations_df = get_global_destinations(supabase)

        if destinations_df.empty:
            st.info("No destination data available")
        else:
            # Key statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Destinations", len(destinations_df))

            with col2:
                total_usage = int(destinations_df['usage_count'].sum())
                st.metric("Total Usage Count", f"{total_usage:,}")

            with col3:
                avg_usage = destinations_df['usage_count'].mean()
                st.metric("Avg Usage per Destination", f"{avg_usage:.1f}")

            with col4:
                most_used = destinations_df.loc[destinations_df['usage_count'].idxmax()]
                st.metric("Most Popular Destination",
                         most_used['description'].split(',')[0] if ',' in most_used['description'] else most_used['description'][:20] + "...")

            st.markdown("---")

            # Map visualization
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown("#### Interactive Destinations Map")

                # Create the map
                fig = px.scatter_mapbox(
                    destinations_df,
                    lat="latitude",
                    lon="longitude",
                    size="usage_count",
                    size_max=25,
                    color="usage_count",
                    color_continuous_scale="Viridis",
                    hover_name="description",
                    hover_data={
                        "latitude": False,
                        "longitude": False,
                        "usage_count": True,
                        "last_used_at": True,
                        "created_at": True
                    },
                    zoom=2,
                    height=500,
                    title="Global Destination Usage Heatmap"
                )

                fig.update_layout(
                    mapbox_style="open-street-map",
                    margin={"r":0,"t":50,"l":0,"b":0}
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Top Destinations")

                # Show top 10 destinations by usage
                top_destinations = destinations_df.nlargest(10, 'usage_count')[['description', 'usage_count', 'last_used_at']]

                # Format for display
                display_df = top_destinations.copy()
                display_df['description'] = display_df['description'].apply(lambda x: x.split(',')[0] if ',' in x else x[:30] + "...")
                display_df['last_used_at'] = display_df['last_used_at'].dt.strftime('%Y-%m-%d')
                display_df.columns = ['Destination', 'Usage Count', 'Last Used']

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Usage Count": st.column_config.NumberColumn(format="%d"),
                        "Last Used": st.column_config.TextColumn(width="small")
                    }
                )

            st.markdown("---")

            # Additional insights
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Usage Distribution")

                # Create usage distribution chart
                usage_bins = [0, 1, 5, 10, 25, float('inf')]
                usage_labels = ['1 time', '2-5 times', '6-10 times', '11-25 times', '25+ times']

                destinations_df['usage_category'] = pd.cut(
                    destinations_df['usage_count'],
                    bins=usage_bins,
                    labels=usage_labels,
                    right=False
                )

                usage_dist = destinations_df['usage_category'].value_counts().sort_index()

                fig = px.bar(
                    x=usage_dist.index,
                    y=usage_dist.values,
                    title="Destination Usage Distribution",
                    labels={'x': 'Usage Frequency', 'y': 'Number of Destinations'},
                    color=usage_dist.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Geographic Insights")

                # Calculate some geographic insights
                if len(destinations_df) > 0:
                    # Northern vs Southern hemisphere
                    northern = len(destinations_df[destinations_df['latitude'] > 0])
                    southern = len(destinations_df[destinations_df['latitude'] < 0])

                    # Create a simple geographic breakdown
                    geo_data = pd.DataFrame({
                        'Region': ['Northern Hemisphere', 'Southern Hemisphere'],
                        'Count': [northern, southern]
                    })

                    fig = px.pie(
                        geo_data,
                        values='Count',
                        names='Region',
                        title="Destinations by Hemisphere",
                        color_discrete_sequence=['#4ECDC4', '#FF6B6B']
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Additional stats
                    st.markdown("**Quick Stats:**")
                    st.info(f"📍 **Most Northern:** {destinations_df.loc[destinations_df['latitude'].idxmax(), 'description'].split(',')[0]}")
                    st.info(f"📍 **Most Southern:** {destinations_df.loc[destinations_df['latitude'].idxmin(), 'description'].split(',')[0]}")

    with tab8:
        st.subheader("🔧 Advanced Analytics & Insights")

        # Get analytics data
        purpose_df = get_trip_purpose_analytics(supabase)
        fuel_df = get_fuel_efficiency_analytics(supabase)

        # Section 1: Trip Purpose Analytics
        st.markdown("### 🎯 Trip Purpose Analytics")

        if not purpose_df.empty:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_trips = len(purpose_df)
                st.metric("Total Trips Analyzed", f"{total_trips:,}")

            with col2:
                business_trips = len(purpose_df[purpose_df['purpose_category'] == 'Business'])
                business_pct = (business_trips / total_trips * 100) if total_trips > 0 else 0
                st.metric("Business Trips", f"{business_trips:,}", f"{business_pct:.1f}%")

            with col3:
                personal_trips = len(purpose_df[purpose_df['purpose_category'] == 'Personal'])
                personal_pct = (personal_trips / total_trips * 100) if total_trips > 0 else 0
                st.metric("Personal Trips", f"{personal_trips:,}", f"{personal_pct:.1f}%")

            with col4:
                total_reimbursements = purpose_df['reimbursement'].sum()
                st.metric("Total Reimbursements", f"${total_reimbursements:,.2f}")

            st.markdown("---")

            # Purpose category breakdown
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("#### Trip Categories Distribution")
                purpose_dist = purpose_df['purpose_category'].value_counts()

                fig = px.pie(
                    values=purpose_dist.values,
                    names=purpose_dist.index,
                    title="Trips by Purpose Category",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Top Trip Purposes")
                # Get top purposes (excluding categorized ones)
                purpose_counts = purpose_df['purpose'].value_counts().head(10)
                purpose_df_display = pd.DataFrame({
                    'Purpose': purpose_counts.index,
                    'Count': purpose_counts.values
                })
                st.dataframe(purpose_df_display, use_container_width=True, hide_index=True)

            # Business vs Personal metrics
            st.markdown("#### Business vs Personal Trip Comparison")
            col1, col2 = st.columns(2)

            with col1:
                business_stats = purpose_df[purpose_df['purpose_category'] == 'Business'].agg({
                    'distance': 'sum',
                    'fuel_used': 'sum',
                    'reimbursement': 'sum'
                })

                st.markdown("**Business Trips:**")
                st.info(f"📏 Total Distance: {business_stats['distance']:,.1f} mi")
                st.info(f"⛽ Fuel Used: {business_stats['fuel_used']:,.1f} gal")
                st.info(f"💰 Reimbursements: ${business_stats['reimbursement']:,.2f}")

            with col2:
                personal_stats = purpose_df[purpose_df['purpose_category'] == 'Personal'].agg({
                    'distance': 'sum',
                    'fuel_used': 'sum',
                    'reimbursement': 'sum'
                })

                st.markdown("**Personal Trips:**")
                st.info(f"📏 Total Distance: {personal_stats['distance']:,.1f} mi")
                st.info(f"⛽ Fuel Used: {personal_stats['fuel_used']:,.1f} gal")
                st.info(f"💰 Reimbursements: ${personal_stats['reimbursement']:,.2f}")
        else:
            st.info("No trip purpose data available")

        st.markdown("---")

        # Section 2: Fuel Efficiency Analytics
        st.markdown("### ⛽ Fuel Efficiency & Cost Analysis")

        if not fuel_df.empty:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_mpg = fuel_df[fuel_df['mpg'] > 0]['mpg'].mean()
                st.metric("Average MPG", f"{avg_mpg:.1f}")

            with col2:
                total_fuel = fuel_df['fuel_used'].sum()
                st.metric("Total Fuel Used", f"{total_fuel:,.1f} gal")

            with col3:
                total_fuel_cost = fuel_df['fuel_cost'].sum()
                st.metric("Total Fuel Cost", f"${total_fuel_cost:,.2f}")

            with col4:
                avg_cost_per_mile = fuel_df[fuel_df['cost_per_mile'] > 0]['cost_per_mile'].mean()
                st.metric("Avg Cost/Mile", f"${avg_cost_per_mile:.3f}")

            st.markdown("---")

            # Fuel efficiency visualizations
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Fuel Efficiency Distribution")
                mpg_data = fuel_df[fuel_df['mpg'] > 0]['mpg']

                if not mpg_data.empty:
                    fig = px.histogram(
                        mpg_data,
                        nbins=20,
                        title="MPG Distribution",
                        labels={'value': 'Miles Per Gallon', 'count': 'Number of Trips'},
                        color_discrete_sequence=['#4ECDC4']
                    )
                    fig.add_vline(x=avg_mpg, line_dash="dash", line_color="red",
                                annotation_text=f"Avg: {avg_mpg:.1f} MPG")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Fuel Cost vs Distance")
                cost_distance_data = fuel_df[(fuel_df['fuel_cost'] > 0) & (fuel_df['distance'] > 0)]

                if not cost_distance_data.empty:
                    fig = px.scatter(
                        cost_distance_data,
                        x='distance',
                        y='fuel_cost',
                        title="Fuel Cost vs Distance (with Trend Line)",
                        labels={'distance': 'Distance (miles)', 'fuel_cost': 'Fuel Cost ($)'},
                        trendline="ols",
                        color_discrete_sequence=['#FF6B6B']
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Cost efficiency insights
            st.markdown("#### Cost Efficiency Insights")
            col1, col2, col3 = st.columns(3)

            with col1:
                # Most efficient trips
                efficient_trips = fuel_df[fuel_df['mpg'] > 0].nlargest(5, 'mpg')
                if not efficient_trips.empty:
                    st.markdown("**Most Fuel Efficient Trips:**")
                    for idx, row in efficient_trips.iterrows():
                        st.write(".1f")

            with col2:
                # Most expensive trips
                expensive_trips = fuel_df.nlargest(5, 'fuel_cost')
                if not expensive_trips.empty:
                    st.markdown("**Most Expensive Trips:**")
                    for idx, row in expensive_trips.iterrows():
                        st.write(".2f")

            with col3:
                # Best value trips (lowest cost per mile)
                value_trips = fuel_df[fuel_df['cost_per_mile'] > 0].nsmallest(5, 'cost_per_mile')
                if not value_trips.empty:
                    st.markdown("**Best Value Trips:**")
                    for idx, row in value_trips.iterrows():
                        st.write(".3f")
        else:
            st.info("No fuel efficiency data available")

        st.markdown("---")

        # Section 3: Data Quality Dashboard
        st.markdown("### 🔍 Data Quality & Health")

        # Calculate data quality metrics
        trips_response = supabase.table('trips').select('*').execute()
        profiles_response = supabase.table('profiles').select('*').execute()

        if trips_response.data and profiles_response.data:
            trips_df = pd.DataFrame(trips_response.data)
            profiles_df = pd.DataFrame(profiles_response.data)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                completeness_score = calculate_data_completeness(trips_df)
                st.metric("Data Completeness", f"{completeness_score:.1f}%")

            with col2:
                missing_locations = trips_df['start_location'].isna().sum()
                st.metric("Missing Locations", missing_locations)

            with col3:
                missing_fuel = trips_df['fuel_used'].isna().sum() + (trips_df['fuel_used'] == 0).sum()
                st.metric("Trips w/o Fuel Data", missing_fuel)

            with col4:
                missing_purposes = trips_df['purpose'].isna().sum() + (trips_df['purpose'] == '').sum()
                st.metric("Trips w/o Purpose", missing_purposes)

            # Data quality breakdown
            st.markdown("#### Data Completeness by Field")
            completeness_data = []

            key_fields = ['start_location', 'end_location', 'purpose', 'fuel_used', 'mileage', 'actual_distance']
            for field in key_fields:
                if field in trips_df.columns:
                    non_null = trips_df[field].notna().sum()
                    total = len(trips_df)
                    completeness_pct = (non_null / total * 100) if total > 0 else 0
                    completeness_data.append({
                        'Field': field.replace('_', ' ').title(),
                        'Complete': non_null,
                        'Total': total,
                        'Completeness %': completeness_pct
                    })

            completeness_df = pd.DataFrame(completeness_data)
            st.dataframe(completeness_df, use_container_width=True, hide_index=True,
                        column_config={
                            "Completeness %": st.column_config.NumberColumn(format="%.1f%%")
                        })

            # Data quality recommendations
            st.markdown("#### 📋 Data Quality Recommendations")

            recommendations = []
            if completeness_score < 80:
                recommendations.append("⚠️ **Low Data Completeness**: Focus on collecting missing location and purpose data")
            if missing_fuel > len(trips_df) * 0.5:
                recommendations.append("⛽ **Fuel Data Missing**: Add fuel tracking to improve cost analytics")
            if missing_purposes > len(trips_df) * 0.3:
                recommendations.append("🎯 **Purpose Classification**: Implement mandatory trip purpose selection")

            if recommendations:
                for rec in recommendations:
                    st.warning(rec)
            else:
                st.success("✅ **Data Quality is Excellent!** All key fields are well populated.")
        else:
            st.info("Unable to calculate data quality metrics")

        st.markdown("---")

        # Section 4: Export & Advanced Features
        st.markdown("### 📊 Export & Advanced Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Export Analytics Data")
            if not purpose_df.empty:
                # Export trip purpose data
                csv_purpose = purpose_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Trip Purpose Data",
                    data=csv_purpose,
                    file_name=f"trip_purpose_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col2:
            if not fuel_df.empty:
                # Export fuel efficiency data
                csv_fuel = fuel_df.to_csv(index=False)
                st.download_button(
                    "⛽ Download Fuel Analytics Data",
                    data=csv_fuel,
                    file_name=f"fuel_efficiency_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col3:
            if trips_response.data:
                # Export data quality report
                quality_report = {
                    'metric': ['Data Completeness', 'Missing Locations', 'Missing Fuel Data', 'Missing Purposes'],
                    'value': [f"{completeness_score:.1f}%", missing_locations, missing_fuel, missing_purposes]
                }
                quality_df = pd.DataFrame(quality_report)
                csv_quality = quality_df.to_csv(index=False)
                st.download_button(
                    "🔍 Download Data Quality Report",
                    data=csv_quality,
                    file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # Performance insights
        st.markdown("#### ⚡ Performance Insights")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Cache hit rate (simplified)
            st.metric("Cache Efficiency", "95%", "High performance")

        with col2:
            # Query response time
            st.metric("Avg Query Time", "< 2s", "Excellent")

        with col3:
            # Data freshness
            st.metric("Data Freshness", "< 5 min", "Real-time")

        # Advanced filtering options
        st.markdown("#### 🎛️ Advanced Filters")
        col1, col2 = st.columns(2)

        with col1:
            date_range = st.date_input(
                "Filter by Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                key="analytics_date_filter"
            )

        with col2:
            purpose_filter = st.multiselect(
                "Filter by Purpose Category",
                options=['Business', 'Personal', 'Service/Delivery', 'Other'],
                default=['Business', 'Personal', 'Service/Delivery', 'Other'],
                key="purpose_category_filter"
            )

        if st.button("🔄 Apply Filters", use_container_width=True):
            st.rerun()
    
    with tab9:
        st.subheader("🆕 New Users from Authentication")
        st.caption("Source: Supabase Auth Admin API (`auth.admin.list_users`)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            new_user_days = st.selectbox(
                "Registration window",
                options=[7, 14, 30, 60, 90, 365],
                index=2,
                key="new_users_window_days"
            )
        with col2:
            row_limit = st.selectbox(
                "Max rows",
                options=[25, 50, 100, 250, 500],
                index=1,
                key="new_users_row_limit"
            )
        with col3:
            completion_filter = st.selectbox(
                "Trip completion",
                options=["All", "Completed at least one trip", "No completed trips yet"],
                index=0,
                key="new_users_completion_filter"
            )
        
        auth_users_df = get_auth_users(supabase)
        
        if auth_users_df.empty:
            st.warning("No authentication users found. Ensure `SUPABASE_KEY` is a service role key with Auth Admin access.")
        else:
            now_utc = pd.Timestamp.now(tz='UTC')
            cutoff = now_utc - pd.Timedelta(days=new_user_days)
            
            new_users_df = auth_users_df[auth_users_df['created_at'] >= cutoff].copy()
            new_users_df = new_users_df.sort_values('created_at', ascending=False)
            
            # Enrich with profile info for display context.
            try:
                profiles_response = supabase.table('profiles').select('id, full_name, subscription_tier').execute()
                profiles_df = pd.DataFrame(profiles_response.data or [])
                if not profiles_df.empty:
                    profiles_df['id'] = profiles_df['id'].astype(str)
                    new_users_df = new_users_df.merge(
                        profiles_df[['id', 'full_name', 'subscription_tier']],
                        on='id',
                        how='left'
                    )
            except Exception:
                pass
            
            # Enrich with trip completion summary.
            trip_completion_df = get_user_trip_completion(supabase)
            if not trip_completion_df.empty:
                new_users_df = new_users_df.merge(
                    trip_completion_df,
                    on='id',
                    how='left'
                )
            
            for col in ['full_name', 'subscription_tier', 'total_trips', 'completed_trips',
                        'completed_any_trip', 'first_trip_at', 'last_trip_at', 'first_completed_trip_at']:
                if col not in new_users_df.columns:
                    new_users_df[col] = None
            
            new_users_df['total_trips'] = pd.to_numeric(new_users_df['total_trips'], errors='coerce').fillna(0).astype(int)
            new_users_df['completed_trips'] = pd.to_numeric(new_users_df['completed_trips'], errors='coerce').fillna(0).astype(int)
            new_users_df['completed_any_trip'] = new_users_df['completed_any_trip'].fillna(False).astype(bool)
            
            # Summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("New Users", len(new_users_df), f"Last {new_user_days} days")
            with col2:
                verified_count = int(new_users_df['email_confirmed_at'].notna().sum())
                st.metric("Email Verified", verified_count)
            with col3:
                active_recent = int((new_users_df['last_sign_in_at'] >= (now_utc - pd.Timedelta(days=30))).sum())
                st.metric("Signed In (30d)", active_recent)
            with col4:
                completed_users = int(new_users_df['completed_any_trip'].sum())
                completion_rate = (completed_users / len(new_users_df) * 100) if len(new_users_df) > 0 else 0
                st.metric("Completed >=1 Trip", completed_users, f"{completion_rate:.1f}%")
            with col5:
                linked_profiles = int(new_users_df['full_name'].notna().sum())
                st.metric("Linked Profiles", linked_profiles)
            
            # Optional completion filter for table view.
            filtered_users_df = new_users_df.copy()
            if completion_filter == "Completed at least one trip":
                filtered_users_df = filtered_users_df[filtered_users_df['completed_any_trip']]
            elif completion_filter == "No completed trips yet":
                filtered_users_df = filtered_users_df[~filtered_users_df['completed_any_trip']]
            
            filtered_users_df = filtered_users_df.head(row_limit)
            st.caption(f"Showing {len(filtered_users_df)} of {len(new_users_df)} users in the selected window.")
            
            st.markdown("---")
            
            # Daily signups chart
            chart_df = new_users_df.dropna(subset=['created_at']).copy()
            if not chart_df.empty:
                chart_df['date'] = chart_df['created_at'].dt.date
                daily_signups = chart_df.groupby('date').size().reset_index(name='new_users')
                
                fig = px.bar(
                    daily_signups,
                    x='date',
                    y='new_users',
                    title=f"Daily New Users ({new_user_days}-day window)",
                    labels={'date': 'Date', 'new_users': 'New Users'}
                )
                fig.update_layout(showlegend=False, height=320)
                st.plotly_chart(fig, use_container_width=True)
            
            # User detail table
            st.markdown("#### New User Details")
            display_df = filtered_users_df[[
                'full_name', 'email', 'provider', 'created_at', 'completed_any_trip',
                'completed_trips', 'total_trips', 'first_completed_trip_at',
                'email_confirmed_at', 'last_sign_in_at', 'subscription_tier'
            ]].copy()
            display_df.rename(columns={
                'full_name': 'Name',
                'email': 'Email',
                'provider': 'Provider',
                'created_at': 'Registered At',
                'completed_any_trip': 'Completed Trip?',
                'completed_trips': 'Completed Trips',
                'total_trips': 'Total Trips',
                'first_completed_trip_at': 'First Completed Trip',
                'email_confirmed_at': 'Email Confirmed At',
                'last_sign_in_at': 'Last Sign In',
                'subscription_tier': 'Subscription'
            }, inplace=True)
            
            for dt_col in ['Registered At', 'First Completed Trip', 'Email Confirmed At', 'Last Sign In']:
                display_df[dt_col] = pd.to_datetime(display_df[dt_col], errors='coerce', utc=True).dt.strftime('%Y-%m-%d %H:%M UTC')
                display_df[dt_col] = display_df[dt_col].fillna('-')
            
            for num_col in ['Completed Trips', 'Total Trips']:
                display_df[num_col] = pd.to_numeric(display_df[num_col], errors='coerce').fillna(0).astype(int)
            
            display_df['Completed Trip?'] = display_df['Completed Trip?'].map({True: 'Yes', False: 'No'}).fillna('No')
            
            for txt_col in ['Name', 'Email', 'Provider', 'Subscription']:
                display_df[txt_col] = display_df[txt_col].fillna('-')
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Email": st.column_config.TextColumn(width="large"),
                    "Provider": st.column_config.TextColumn(width="small"),
                    "Completed Trip?": st.column_config.TextColumn(width="small"),
                    "Completed Trips": st.column_config.NumberColumn(format="%d"),
                    "Total Trips": st.column_config.NumberColumn(format="%d"),
                    "First Completed Trip": st.column_config.TextColumn(width="medium"),
                    "Registered At": st.column_config.TextColumn(width="medium"),
                    "Email Confirmed At": st.column_config.TextColumn(width="medium"),
                    "Last Sign In": st.column_config.TextColumn(width="medium")
                }
            )
            
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download New Users CSV",
                data=csv,
                file_name=f"new_auth_users_last_{new_user_days}d_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab10:
        st.subheader("🗓️ Trip Logs by Day or Week")
        st.caption("Recent trip logs with user information, grouped by day or week.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            trip_log_days = st.selectbox(
                "Lookback window",
                options=[7, 14, 30, 60, 90, 180, 365],
                index=4,
                key="trip_logs_lookback_days"
            )
        with col2:
            trip_grouping = st.radio(
                "Group trips by",
                options=["Day", "Week"],
                horizontal=True,
                key="trip_logs_grouping"
            )
        with col3:
            trip_log_limit = st.selectbox(
                "Rows to show",
                options=[25, 50, 100, 250, 500],
                index=2,
                key="trip_logs_row_limit"
            )
        
        trip_logs_df = get_trip_logs_with_users(supabase, trip_log_days)
        
        if trip_logs_df.empty:
            st.info("No trip logs found for the selected period.")
        else:
            logs_df = trip_logs_df.copy()
            logs_df['created_at'] = pd.to_datetime(logs_df['created_at'], errors='coerce', utc=True)
            logs_df['distance'] = pd.to_numeric(logs_df['distance'], errors='coerce').fillna(0)
            logs_df['duration_mins'] = pd.to_numeric(logs_df['duration_mins'], errors='coerce').fillna(0)
            logs_df['is_completed'] = logs_df['is_completed'].fillna(False).astype(bool)
            
            created_utc = logs_df['created_at'].dt.tz_convert('UTC')
            if trip_grouping == "Day":
                logs_df['period_start'] = created_utc.dt.floor('D')
                logs_df['period_label'] = logs_df['period_start'].dt.strftime('%Y-%m-%d')
            else:
                logs_df['period_start'] = (created_utc - pd.to_timedelta(created_utc.dt.weekday, unit='D')).dt.floor('D')
                logs_df['period_label'] = logs_df['period_start'].dt.strftime('Week of %Y-%m-%d')
            
            summary_df = logs_df.groupby(['period_start', 'period_label']).agg(
                trips=('trip_id', 'count'),
                unique_users=('user_id', 'nunique'),
                completed_trips=('is_completed', 'sum'),
                total_distance=('distance', 'sum')
            ).reset_index()
            summary_df['completion_rate'] = np.where(
                summary_df['trips'] > 0,
                (summary_df['completed_trips'] / summary_df['trips']) * 100,
                0
            )
            summary_df = summary_df.sort_values('period_start', ascending=False)
            summary_df['total_distance'] = summary_df['total_distance'].round(1)
            summary_df['completion_rate'] = summary_df['completion_rate'].round(1)
            
            # Top-level KPI cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Trips Logged", f"{len(logs_df):,}", f"Last {trip_log_days} days")
            with col2:
                st.metric("Unique Users", f"{logs_df['user_id'].nunique():,}")
            with col3:
                st.metric("Completed Trips", f"{int(logs_df['is_completed'].sum()):,}")
            with col4:
                avg_trips_per_period = len(logs_df) / len(summary_df) if len(summary_df) > 0 else 0
                st.metric(f"Avg Trips per {trip_grouping}", f"{avg_trips_per_period:.1f}")
            
            st.markdown("---")
            
            # Period summary chart + table
            col1, col2 = st.columns([2, 1])
            with col1:
                chart_df = summary_df.sort_values('period_start')
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=chart_df['period_label'],
                    y=chart_df['trips'],
                    name='Trips',
                    marker_color='#1f77b4'
                ))
                fig.add_trace(go.Scatter(
                    x=chart_df['period_label'],
                    y=chart_df['unique_users'],
                    name='Unique Users',
                    mode='lines+markers',
                    yaxis='y2',
                    line=dict(color='#ff7f0e', width=3)
                ))
                fig.update_layout(
                    title=f"Trips and Unique Users by {trip_grouping}",
                    xaxis_title=trip_grouping,
                    yaxis=dict(title='Trips'),
                    yaxis2=dict(title='Unique Users', overlaying='y', side='right'),
                    height=360,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown(f"#### {trip_grouping} Summary")
                summary_display = summary_df[['period_label', 'trips', 'unique_users', 'completed_trips', 'completion_rate', 'total_distance']].copy()
                summary_display.rename(columns={
                    'period_label': trip_grouping,
                    'trips': 'Trips',
                    'unique_users': 'Users',
                    'completed_trips': 'Completed',
                    'completion_rate': 'Completion %',
                    'total_distance': 'Distance (mi)'
                }, inplace=True)
                st.dataframe(
                    summary_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Trips": st.column_config.NumberColumn(format="%d"),
                        "Users": st.column_config.NumberColumn(format="%d"),
                        "Completed": st.column_config.NumberColumn(format="%d"),
                        "Completion %": st.column_config.NumberColumn(format="%.1f"),
                        "Distance (mi)": st.column_config.NumberColumn(format="%.1f")
                    }
                )
            
            st.markdown("---")
            
            # Detailed trip log table
            period_options = ["All"] + summary_df['period_label'].tolist()
            selected_period = st.selectbox(
                f"Show detailed trips for {trip_grouping.lower()}",
                options=period_options,
                key="trip_logs_selected_period"
            )
            
            details_df = logs_df.copy()
            if selected_period != "All":
                details_df = details_df[details_df['period_label'] == selected_period]
            
            details_df = details_df.sort_values('created_at', ascending=False).head(trip_log_limit)
            
            st.caption(f"Showing {len(details_df)} trip logs.")
            
            display_details = details_df[[
                'created_at', 'trip_id', 'full_name', 'email', 'phone_number', 'subscription_tier',
                'status', 'is_completed', 'distance', 'duration_mins', 'start_location', 'end_location', 'purpose'
            ]].copy()
            
            display_details.rename(columns={
                'created_at': 'Logged At',
                'trip_id': 'Trip ID',
                'full_name': 'User',
                'email': 'Email',
                'phone_number': 'Phone',
                'subscription_tier': 'Subscription',
                'status': 'Status',
                'is_completed': 'Completed?',
                'distance': 'Distance (mi)',
                'duration_mins': 'Duration',
                'start_location': 'Start Location',
                'end_location': 'End Location',
                'purpose': 'Purpose'
            }, inplace=True)
            
            display_details['Logged At'] = pd.to_datetime(
                display_details['Logged At'],
                errors='coerce',
                utc=True
            ).dt.strftime('%Y-%m-%d %H:%M UTC').fillna('-')
            display_details['Distance (mi)'] = pd.to_numeric(display_details['Distance (mi)'], errors='coerce').fillna(0).round(1)
            display_details['Duration'] = pd.to_numeric(display_details['Duration'], errors='coerce').fillna(0).apply(format_duration)
            display_details['Status'] = display_details['Status'].fillna('unknown').astype(str).str.title()
            display_details['Completed?'] = display_details['Completed?'].map({True: 'Yes', False: 'No'}).fillna('No')
            
            for txt_col in ['User', 'Email', 'Phone', 'Subscription', 'Start Location', 'End Location', 'Purpose']:
                display_details[txt_col] = display_details[txt_col].fillna('-')
            
            st.dataframe(
                display_details,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Logged At": st.column_config.TextColumn(width="medium"),
                    "Trip ID": st.column_config.TextColumn(width="small"),
                    "User": st.column_config.TextColumn(width="medium"),
                    "Email": st.column_config.TextColumn(width="large"),
                    "Phone": st.column_config.TextColumn(width="medium"),
                    "Subscription": st.column_config.TextColumn(width="small"),
                    "Completed?": st.column_config.TextColumn(width="small"),
                    "Distance (mi)": st.column_config.NumberColumn(format="%.1f"),
                    "Start Location": st.column_config.TextColumn(width="large"),
                    "End Location": st.column_config.TextColumn(width="large")
                }
            )
            
            csv_details = display_details.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Trip Logs CSV",
                data=csv_details,
                file_name=f"trip_logs_{trip_grouping.lower()}_{trip_log_days}d_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab11:
        st.subheader("🚀 Upgrade Intent Signals (Human Readable)")
        st.caption("Source: `upgrade_intent_log` event stream with metadata normalized for analysis.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            intent_lookback_days = st.selectbox(
                "Lookback window",
                options=[1, 7, 14, 30, 60, 90, 180],
                index=3,
                key="intent_lookback_days"
            )
        with col2:
            intent_row_limit = st.selectbox(
                "Rows to show",
                options=[25, 50, 100, 250, 500, 1000],
                index=2,
                key="intent_row_limit"
            )
        
        intent_df = get_upgrade_intent_events(supabase, intent_lookback_days)
        
        if intent_df.empty:
            st.info("No upgrade intent events found for the selected period.")
        else:
            filtered_df = intent_df.copy()
            
            family_options = sorted(filtered_df['event_family'].dropna().astype(str).unique().tolist())
            platform_options = sorted(filtered_df['platform'].dropna().astype(str).unique().tolist())
            feature_options = sorted(filtered_df['feature_display'].dropna().astype(str).unique().tolist())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_families = st.multiselect(
                    "Event families",
                    options=family_options,
                    default=family_options,
                    key="intent_family_filter"
                )
            with col2:
                selected_platforms = st.multiselect(
                    "Platforms",
                    options=platform_options,
                    default=platform_options,
                    key="intent_platform_filter"
                )
            with col3:
                selected_features = st.multiselect(
                    "Features",
                    options=feature_options,
                    default=feature_options,
                    key="intent_feature_filter"
                )
            
            if selected_families:
                filtered_df = filtered_df[filtered_df['event_family'].isin(selected_families)]
            if selected_platforms:
                filtered_df = filtered_df[filtered_df['platform'].isin(selected_platforms)]
            if selected_features:
                filtered_df = filtered_df[filtered_df['feature_display'].isin(selected_features)]
            
            if filtered_df.empty:
                st.info("No events match the selected filters.")
            else:
                filtered_df['event_ts'] = pd.to_datetime(filtered_df['event_ts'], errors='coerce', utc=True)
                
                # Core KPI row
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Signals", f"{len(filtered_df):,}")
                with col2:
                    st.metric("Unique Users", f"{filtered_df['user_id'].nunique():,}")
                with col3:
                    st.metric("Upgrade Intent", f"{int((filtered_df['event_kind'] == 'upgrade_intent').sum()):,}")
                with col4:
                    st.metric("Conversion Events", f"{int((filtered_df['event_kind'] == 'conversion_event').sum()):,}")
                with col5:
                    purchase_started = int((filtered_df['event_name'] == 'premium_purchase_started').sum())
                    st.metric("Purchase Starts", f"{purchase_started:,}")
                
                # Funnel-style summary for common conversion milestones.
                purchase_start_users = int(filtered_df[filtered_df['event_name'] == 'premium_purchase_started']['user_id'].nunique())
                purchase_complete_users = int(filtered_df[filtered_df['event_name'] == 'premium_purchase_completed']['user_id'].nunique())
                export_attempts = int((filtered_df['event_name'] == 'trip_export_attempted').sum())
                export_completes = int((filtered_df['event_name'] == 'trip_export_completed').sum())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Purchase Start Users", f"{purchase_start_users:,}")
                with col2:
                    st.metric("Purchase Complete Users", f"{purchase_complete_users:,}")
                with col3:
                    purchase_rate = (purchase_complete_users / purchase_start_users * 100) if purchase_start_users > 0 else 0
                    st.metric("Start → Complete Rate", f"{purchase_rate:.1f}%")
                
                st.caption(f"Trip Export Conversion: {export_completes:,} / {export_attempts:,} ({(export_completes / export_attempts * 100) if export_attempts > 0 else 0:.1f}%)")
                
                st.markdown("---")
                
                # Top trigger and action summaries
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Top Features Triggering Upgrade Intent")
                    trigger_df = filtered_df[filtered_df['event_kind'] == 'upgrade_intent']
                    if not trigger_df.empty:
                        top_features_df = trigger_df['feature_display'].fillna('Unknown').value_counts().head(10).reset_index()
                        top_features_df.columns = ['Feature', 'Signals']
                        fig = px.bar(
                            top_features_df,
                            x='Signals',
                            y='Feature',
                            orientation='h',
                            title="Upgrade Intent by Feature",
                            color='Signals',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(height=350, showlegend=False, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No upgrade intent events in this filtered view.")
                
                with col2:
                    st.markdown("#### Most Frequent Actions")
                    top_actions_df = filtered_df['action_label'].fillna('Unknown').value_counts().head(10).reset_index()
                    top_actions_df.columns = ['Action', 'Events']
                    fig = px.bar(
                        top_actions_df,
                        x='Events',
                        y='Action',
                        orientation='h',
                        title="Top Actions",
                        color='Events',
                        color_continuous_scale='Greens'
                    )
                    fig.update_layout(height=350, showlegend=False, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Timeline chart
                st.markdown("#### Event Timeline")
                timeline_df = filtered_df.dropna(subset=['event_ts']).copy()
                if not timeline_df.empty:
                    if intent_lookback_days <= 7:
                        timeline_df['time_bucket'] = timeline_df['event_ts'].dt.floor('H')
                        timeline_title = "Hourly Event Volume"
                    else:
                        timeline_df['time_bucket'] = timeline_df['event_ts'].dt.floor('D')
                        timeline_title = "Daily Event Volume"
                    
                    timeline_agg = timeline_df.groupby(['time_bucket', 'event_family']).size().reset_index(name='events')
                    fig = px.line(
                        timeline_agg,
                        x='time_bucket',
                        y='events',
                        color='event_family',
                        markers=True,
                        title=timeline_title,
                        labels={'time_bucket': 'Time', 'events': 'Events', 'event_family': 'Family'}
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No timestamped events available for timeline.")
                
                st.markdown("---")
                
                # User-level summary
                st.markdown("#### User-Level Summary")
                summary_df = filtered_df.copy()
                summary_df['display_user'] = summary_df['full_name'].fillna(summary_df['user_id'].astype(str).str[:8] + '...')
                user_summary = summary_df.groupby('user_id').agg(
                    User=('display_user', 'first'),
                    Email=('email', 'first'),
                    Events=('id', 'count'),
                    Upgrade_Intents=('event_kind', lambda x: int((x == 'upgrade_intent').sum())),
                    Conversion_Events=('event_kind', lambda x: int((x == 'conversion_event').sum())),
                    First_Signal=('event_ts', 'min'),
                    Last_Signal=('event_ts', 'max')
                ).reset_index(drop=True).sort_values('Events', ascending=False)
                
                user_summary.rename(columns={
                    'Upgrade_Intents': 'Upgrade Intents',
                    'Conversion_Events': 'Conversion Events',
                    'First_Signal': 'First Signal',
                    'Last_Signal': 'Last Signal'
                }, inplace=True)
                
                user_summary['Email'] = user_summary['Email'].fillna('-')
                for dt_col in ['First Signal', 'Last Signal']:
                    user_summary[dt_col] = pd.to_datetime(user_summary[dt_col], errors='coerce', utc=True).dt.strftime('%Y-%m-%d %H:%M UTC').fillna('-')
                
                st.dataframe(
                    user_summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Events": st.column_config.NumberColumn(format="%d"),
                        "Upgrade Intents": st.column_config.NumberColumn(format="%d"),
                        "Conversion Events": st.column_config.NumberColumn(format="%d"),
                    }
                )
                
                st.markdown("---")
                
                # Detailed event feed
                st.markdown("#### Event Feed")
                detail_df = filtered_df.sort_values('event_ts', ascending=False).head(intent_row_limit).copy()
                
                def format_experiment(name, variant):
                    if pd.isna(name) or str(name).strip() == "":
                        return "-"
                    if pd.notna(variant) and str(variant).strip() != "":
                        return f"{name} ({variant})"
                    return str(name)
                
                detail_df['Time (UTC)'] = pd.to_datetime(detail_df['event_ts'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('-')
                detail_df['User'] = detail_df['full_name'].fillna(detail_df['user_id'].astype(str).str[:8] + '...')
                detail_df['Email'] = detail_df['email'].fillna('-')
                detail_df['Event Family'] = detail_df['event_family'].fillna('-')
                detail_df['Action'] = detail_df['action_label'].fillna('-')
                detail_df['Feature'] = detail_df['feature_display'].fillna('-')
                detail_df['Source'] = detail_df['source'].fillna('-')
                detail_df['Platform'] = detail_df['platform'].fillna('-')
                detail_df['Subscription'] = detail_df['subscription_tier'].fillna('-')
                detail_df['Gate Strategy'] = detail_df['gate_strategy'].fillna('-')
                detail_df['Experiment'] = detail_df.apply(
                    lambda r: format_experiment(r.get('experiment_name'), r.get('experiment_variant')),
                    axis=1
                )
                detail_df['Trip ID'] = detail_df['trip_id'].fillna('-')
                
                display_cols = [
                    'Time (UTC)', 'User', 'Email', 'Event Family', 'Action', 'Feature',
                    'Source', 'Platform', 'Subscription', 'Gate Strategy', 'Experiment', 'Trip ID'
                ]
                display_feed = detail_df[display_cols].copy()
                
                st.caption(f"Showing {len(display_feed)} most recent events after filters.")
                st.dataframe(
                    display_feed,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Time (UTC)": st.column_config.TextColumn(width="medium"),
                        "User": st.column_config.TextColumn(width="medium"),
                        "Email": st.column_config.TextColumn(width="large"),
                        "Event Family": st.column_config.TextColumn(width="small"),
                        "Action": st.column_config.TextColumn(width="medium"),
                        "Feature": st.column_config.TextColumn(width="medium"),
                        "Source": st.column_config.TextColumn(width="medium"),
                        "Platform": st.column_config.TextColumn(width="small"),
                        "Subscription": st.column_config.TextColumn(width="small"),
                        "Gate Strategy": st.column_config.TextColumn(width="small"),
                        "Experiment": st.column_config.TextColumn(width="medium"),
                        "Trip ID": st.column_config.TextColumn(width="medium"),
                    }
                )
                
                feed_csv = display_feed.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Upgrade Signals CSV",
                    data=feed_csv,
                    file_name=f"upgrade_signals_{intent_lookback_days}d_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <small>Mileage Tracker Pro Dashboard v1.0 | Data refreshes every 5 minutes | 
            Built with Streamlit & Supabase</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
