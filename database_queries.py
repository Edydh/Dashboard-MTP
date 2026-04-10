"""
Advanced database queries for Mileage Tracker Pro Dashboard
This module is not yet wired into dashboard.py.
It has been partially aligned with shared dashboard helpers, but several
methods still rely on full-table reads and should be audited before reuse.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict
import streamlit as st

from dashboard_data import calculate_trip_duration_minutes, normalize_trip_distance
from supabase_client import Client

class DatabaseQueries:
    """Class to handle all database queries with optimized SQL"""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
    
    @st.cache_data(ttl=300)
    def get_user_summary_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive user statistics using raw SQL
        """
        try:
            # Using Supabase's RPC function capability for complex queries
            # Note: You'll need to create these functions in your Supabase SQL editor
            
            _sql_template = """
            SELECT 
                COUNT(DISTINCT p.id) as total_users,
                COUNT(DISTINCT CASE 
                    WHEN t.created_at >= NOW() - INTERVAL '30 days' 
                    THEN t.user_id 
                END) as active_users_30d,
                COUNT(DISTINCT CASE 
                    WHEN t.created_at >= NOW() - INTERVAL '7 days' 
                    THEN t.user_id 
                END) as active_users_7d,
                COUNT(DISTINCT CASE 
                    WHEN t.created_at >= NOW() - INTERVAL '1 day' 
                    THEN t.user_id 
                END) as active_users_today
            FROM profiles p
            LEFT JOIN trips t ON p.id = t.user_id
            """
            
            # For now, we'll use the API approach since direct SQL requires RPC functions.
            # `_sql_template` is kept as a reference for future RPC/database-side implementation.
            users = self.client.table('profiles').select('id').execute()
            total_users = len(users.data) if users.data else 0
            
            # Get active users for different periods
            now = datetime.now()
            periods = {
                'today': 1,
                '7d': 7,
                '30d': 30
            }
            
            stats = {'total_users': total_users}
            
            for period_name, days in periods.items():
                cutoff = (now - timedelta(days=days)).isoformat()
                trips = self.client.table('trips').select('user_id').gte('created_at', cutoff).execute()
                if trips.data:
                    active_users = len(set(trip['user_id'] for trip in trips.data))
                    stats[f'active_users_{period_name}'] = active_users
                else:
                    stats[f'active_users_{period_name}'] = 0
            
            return stats
            
        except Exception as e:
            st.error(f"Error in get_user_summary_stats: {str(e)}")
            return {}
    
    @st.cache_data(ttl=300)
    def get_user_cohort_analysis(self, cohort_size: int = 7) -> pd.DataFrame:
        """
        Perform cohort analysis on user retention
        """
        try:
            # Get all profiles with their registration date
            users = self.client.table('profiles').select('id, created_at').execute()
            users_df = pd.DataFrame(users.data)
            users_df['created_at'] = pd.to_datetime(users_df['created_at'])
            users_df['cohort_week'] = users_df['created_at'].dt.to_period('W')
            
            # Get all trips
            trips = self.client.table('trips').select('user_id, created_at').execute()
            trips_df = pd.DataFrame(trips.data)
            
            if trips_df.empty:
                return pd.DataFrame()
            
            trips_df['created_at'] = pd.to_datetime(trips_df['created_at'])
            trips_df['activity_week'] = trips_df['created_at'].dt.to_period('W')
            
            # Merge to get cohort information
            cohort_data = trips_df.merge(users_df[['id', 'cohort_week']], 
                                        left_on='user_id', right_on='id')
            
            # Calculate weeks since registration
            cohort_data['weeks_since_registration'] = (
                cohort_data['activity_week'] - cohort_data['cohort_week']
            ).apply(lambda x: x.n if hasattr(x, 'n') else 0)
            
            # Create cohort table
            cohort_table = cohort_data.groupby(['cohort_week', 'weeks_since_registration'])\
                                     ['user_id'].nunique().reset_index()
            
            # Pivot for visualization
            cohort_pivot = cohort_table.pivot(index='cohort_week', 
                                             columns='weeks_since_registration', 
                                             values='user_id').fillna(0)
            
            # Calculate retention rates
            cohort_sizes = cohort_data.groupby('cohort_week')['user_id'].nunique()
            retention_table = cohort_pivot.divide(cohort_sizes, axis=0) * 100
            
            return retention_table
            
        except Exception as e:
            st.error(f"Error in cohort analysis: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=300)
    def get_trip_patterns(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze trip patterns and user behavior
        """
        try:
            # Get trips data
            trips = self.client.table('trips').select('*').execute()
            
            if not trips.data:
                return {}
            
            trips_df = pd.DataFrame(trips.data)
            trips_df['created_at'] = pd.to_datetime(trips_df['created_at'])
            trips_df['distance'] = normalize_trip_distance(
                trips_df.get('actual_distance'),
                trips_df.get('mileage')
            )
            if {'start_time', 'end_time'}.issubset(trips_df.columns):
                trips_df['duration'] = calculate_trip_duration_minutes(
                    trips_df['start_time'],
                    trips_df['end_time']
                ).fillna(0)
            elif 'duration' in trips_df.columns:
                trips_df['duration'] = pd.to_numeric(
                    trips_df['duration'], errors='coerce'
                ).fillna(0)
            else:
                trips_df['duration'] = pd.Series(0.0, index=trips_df.index)
            
            # Time-based patterns
            trips_df['hour'] = trips_df['created_at'].dt.hour
            trips_df['day_of_week'] = trips_df['created_at'].dt.day_name()
            trips_df['month'] = trips_df['created_at'].dt.month_name()
            trips_df['date'] = trips_df['created_at'].dt.date
            
            patterns = {
                'hourly': trips_df.groupby('hour').agg({
                    'id': 'count',
                    'distance': 'mean',
                    'duration': 'mean'
                }).rename(columns={'id': 'trip_count'}),
                
                'daily': trips_df.groupby('day_of_week').agg({
                    'id': 'count',
                    'distance': 'sum',
                    'duration': 'sum',
                    'user_id': 'nunique'
                }).rename(columns={'id': 'trip_count', 'user_id': 'unique_users'}),
                
                'monthly_trend': trips_df.groupby(trips_df['created_at'].dt.to_period('M')).agg({
                    'id': 'count',
                    'distance': 'sum',
                    'user_id': 'nunique'
                }).rename(columns={'id': 'trip_count', 'user_id': 'unique_users'}),
                
                'user_frequency': trips_df.groupby('user_id').agg({
                    'id': 'count',
                    'distance': 'sum',
                    'duration': 'sum',
                    'created_at': ['min', 'max']
                }).rename(columns={'id': 'trip_count'})
            }
            
            return patterns
            
        except Exception as e:
            st.error(f"Error in trip patterns analysis: {str(e)}")
            return {}
    
    @st.cache_data(ttl=300)
    def get_user_segments(self) -> pd.DataFrame:
        """
        Segment users based on their activity patterns
        """
        try:
            # Get user trip statistics
            trips = self.client.table('trips').select('user_id, mileage, actual_distance, start_time, end_time, created_at').execute()
            
            if not trips.data:
                return pd.DataFrame()
            
            trips_df = pd.DataFrame(trips.data)
            trips_df['created_at'] = pd.to_datetime(trips_df['created_at'])
            
            # Calculate duration from start_time and end_time if available
            if 'start_time' in trips_df.columns and 'end_time' in trips_df.columns:
                trips_df['duration'] = calculate_trip_duration_minutes(
                    trips_df['start_time'],
                    trips_df['end_time']
                ).fillna(0)
            else:
                trips_df['duration'] = 0
            
            # Use actual_distance if available, otherwise use mileage
            trips_df['distance'] = normalize_trip_distance(
                trips_df.get('actual_distance'),
                trips_df.get('mileage')
            )
            
            # Calculate user metrics
            now = datetime.now()
            user_metrics = trips_df.groupby('user_id').agg({
                'user_id': 'count',  # trip count
                'distance': ['sum', 'mean'],
                'duration': ['sum', 'mean'],
                'created_at': lambda x: (now - x.max()).days  # days since last trip
            })
            
            user_metrics.columns = ['trip_count', 'total_distance', 'avg_distance', 
                                   'total_duration', 'avg_duration', 'days_inactive']
            
            # Calculate activity score
            user_metrics['activity_score'] = (
                user_metrics['trip_count'] * 0.4 +
                (user_metrics['total_distance'] / user_metrics['total_distance'].max() * 100) * 0.3 +
                (100 - user_metrics['days_inactive'].clip(upper=100)) * 0.3
            )
            
            # Segment users
            def segment_user(row):
                if row['days_inactive'] > 30:
                    return 'Churned'
                elif row['days_inactive'] > 14:
                    return 'At Risk'
                elif row['trip_count'] >= 20 and row['days_inactive'] <= 7:
                    return 'Power User'
                elif row['trip_count'] >= 10:
                    return 'Regular'
                elif row['trip_count'] >= 5:
                    return 'Casual'
                else:
                    return 'New'
            
            user_metrics['segment'] = user_metrics.apply(segment_user, axis=1)
            
            return user_metrics.reset_index()
            
        except Exception as e:
            st.error(f"Error in user segmentation: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=300)
    def get_predictive_metrics(self) -> Dict[str, Any]:
        """
        Calculate predictive metrics for churn and growth
        """
        try:
            # Get recent activity data
            cutoff_30d = (datetime.now() - timedelta(days=30)).isoformat()
            cutoff_7d = (datetime.now() - timedelta(days=7)).isoformat()
            
            trips_30d = self.client.table('trips').select('user_id, created_at').gte('created_at', cutoff_30d).execute()
            trips_7d = self.client.table('trips').select('user_id, created_at').gte('created_at', cutoff_7d).execute()
            
            if not trips_30d.data:
                return {}
            
            # Calculate trends
            users_30d = set(trip['user_id'] for trip in trips_30d.data)
            users_7d = set(trip['user_id'] for trip in trips_7d.data)
            
            # Churn risk: users active in 30d but not in 7d
            at_risk_users = users_30d - users_7d
            churn_risk_rate = len(at_risk_users) / len(users_30d) * 100 if users_30d else 0
            
            # Growth trend
            trips_df = pd.DataFrame(trips_30d.data)
            trips_df['created_at'] = pd.to_datetime(trips_df['created_at'])
            trips_df['week'] = trips_df['created_at'].dt.isocalendar().week
            
            weekly_trips = trips_df.groupby('week').size()
            if len(weekly_trips) > 1:
                growth_trend = (weekly_trips.iloc[-1] - weekly_trips.iloc[-2]) / weekly_trips.iloc[-2] * 100
            else:
                growth_trend = 0
            
            # Engagement score
            avg_trips_per_user = len(trips_30d.data) / len(users_30d) if users_30d else 0
            
            metrics = {
                'churn_risk_rate': churn_risk_rate,
                'users_at_risk': len(at_risk_users),
                'weekly_growth_trend': growth_trend,
                'engagement_score': min(100, avg_trips_per_user * 10),
                'predicted_active_next_week': int(len(users_7d) * (1 + growth_trend/100))
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"Error in predictive metrics: {str(e)}")
            return {}

# SQL function templates for Supabase (to be created in Supabase SQL editor)
SQL_FUNCTIONS = """
-- Create these functions in your Supabase SQL editor for better performance

-- Function to get user statistics
CREATE OR REPLACE FUNCTION get_user_stats(days_back INTEGER DEFAULT 30)
RETURNS TABLE (
    total_users BIGINT,
    active_users BIGINT,
    new_users BIGINT,
    churned_users BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(DISTINCT p.id) as total_users,
        COUNT(DISTINCT CASE 
            WHEN t.created_at >= NOW() - INTERVAL '1 day' * days_back 
            THEN t.user_id 
        END) as active_users,
        COUNT(DISTINCT CASE 
            WHEN p.created_at >= NOW() - INTERVAL '1 day' * days_back 
            THEN p.id 
        END) as new_users,
        COUNT(DISTINCT CASE 
            WHEN t.user_id IS NULL 
            OR MAX(t.created_at) < NOW() - INTERVAL '1 day' * days_back 
            THEN p.id 
        END) as churned_users
    FROM profiles p
    LEFT JOIN trips t ON p.id = t.user_id
    GROUP BY p.id;
END;
$$ LANGUAGE plpgsql;

-- Function to get top active users with details
CREATE OR REPLACE FUNCTION get_top_active_users(limit_count INTEGER DEFAULT 10)
RETURNS TABLE (
    user_id UUID,
    full_name TEXT,
    phone_number TEXT,
    subscription_tier TEXT,
    trip_count BIGINT,
    total_distance NUMERIC,
    total_duration NUMERIC,
    last_trip_date TIMESTAMP,
    days_since_last_trip INTEGER,
    trips_per_week NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH user_stats AS (
        SELECT 
            p.id,
            p.full_name,
            p.phone_number,
            p.subscription_tier,
            COUNT(t.id) as trip_count,
            COALESCE(SUM(t.distance), 0) as total_distance,
            COALESCE(SUM(t.duration), 0) as total_duration,
            MAX(t.created_at) as last_trip_date,
            EXTRACT(DAY FROM NOW() - MAX(t.created_at))::INTEGER as days_since_last_trip,
            COUNT(t.id)::NUMERIC / GREATEST(1, EXTRACT(WEEK FROM NOW() - MIN(t.created_at))) as trips_per_week
        FROM profiles p
        LEFT JOIN trips t ON p.id = t.user_id
        GROUP BY p.id, p.full_name, p.phone_number, p.subscription_tier
    )
    SELECT * FROM user_stats
    ORDER BY trip_count DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function for cohort analysis
CREATE OR REPLACE FUNCTION get_cohort_retention()
RETURNS TABLE (
    cohort_month DATE,
    month_number INTEGER,
    users_count BIGINT,
    retention_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH     cohorts AS (
        SELECT 
            DATE_TRUNC('month', p.created_at) as cohort_month,
            p.id as user_id
        FROM profiles p
    ),
    activities AS (
        SELECT 
            c.cohort_month,
            c.user_id,
            DATE_TRUNC('month', t.created_at) as activity_month,
            EXTRACT(MONTH FROM AGE(DATE_TRUNC('month', t.created_at), c.cohort_month))::INTEGER as month_number
        FROM cohorts c
        JOIN trips t ON c.user_id = t.user_id
    ),
    cohort_sizes AS (
        SELECT cohort_month, COUNT(DISTINCT user_id) as cohort_size
        FROM cohorts
        GROUP BY cohort_month
    )
    SELECT 
        a.cohort_month,
        a.month_number,
        COUNT(DISTINCT a.user_id) as users_count,
        COUNT(DISTINCT a.user_id)::NUMERIC / cs.cohort_size * 100 as retention_rate
    FROM activities a
    JOIN cohort_sizes cs ON a.cohort_month = cs.cohort_month
    GROUP BY a.cohort_month, a.month_number, cs.cohort_size
    ORDER BY a.cohort_month, a.month_number;
END;
$$ LANGUAGE plpgsql;
"""
