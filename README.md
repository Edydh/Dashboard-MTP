# 🚗 Mileage Tracker Pro Dashboard

A comprehensive real-time analytics dashboard for monitoring user activity, trip statistics, revenue metrics, and growth trends for the Mileage Tracker Pro application.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-red)
![Supabase](https://img.shields.io/badge/Supabase-2.7.4-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📊 Features

### Dashboard Tabs

The current dashboard exposes 12 tabs. The main views are summarized below.

#### 1. **📊 Overview**
- Total users count with active user metrics
- Top 10 most active users with detailed statistics
- Real-time activity monitoring
- Trip statistics and trends

#### 2. **💰 Revenue Analytics**
- Monthly Recurring Revenue (MRR) tracking
- Annual Recurring Revenue (ARR) projections
- Average Revenue Per User (ARPU) calculations
- Subscription tier distribution visualization
- Revenue breakdown by subscription tier

#### 3. **📈 Growth Metrics**
- Week-over-Week (WoW) growth tracking
- Month-over-Month (MoM) analysis
- Quarter-over-Quarter (QoQ) trends
- Daily signup charts
- Cumulative user growth visualization

#### 4. **🔴 Live Activity Feed**
- Real-time trip monitoring
- Last hour activity metrics
- Active users tracking
- Trip timeline visualization
- Configurable display (10, 20, 50, 100 trips)

#### 5. **👥 User Analytics**
- Detailed user behavior analysis
- Users at risk identification
- Activity patterns and heatmaps
- Trip duration and distance analytics
- CSV export functionality

#### 6. **📈 Retention Analysis**
- Day 1, Day 7, Day 30 retention metrics
- User activation funnel
- Time-to-first-trip distribution
- Weekly cohort analysis
- Smart recommendations based on retention data

#### Additional Current Views
- **💸 Expenses**: Expense totals, reimbursements, and related operational analytics
- **🗺️ Destinations Map**: Destination and trip-location views
- **🔧 Advanced Analytics**: Deeper derived metrics and supporting analysis
- **🆕 New Users**: Recent signup and early-activation analysis
- **🗓️ Trip Logs**: Day/week trip logs with user detail
- **🚀 Upgrade Signals**: Paywall, purchase funnel, and tracked feature-usage views

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Supabase account with configured database
- macOS/Linux/Windows with terminal access

### Installation

#### Option 1: Quick Setup Script
```bash
# Clone the repository
git clone https://github.com/yourusername/Dashboard-MTP.git
cd Dashboard-MTP

# Run the quick setup script
chmod +x quick_setup.sh
./quick_setup.sh
```

#### Option 2: Manual Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Dashboard-MTP.git
cd Dashboard-MTP

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Create a `.env` file in the project root:
```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key
```

2. Use a Supabase service role key, not an anon key.
   The dashboard reads the Supabase Auth Admin API via `auth.admin.list_users()`, so anon/public keys are not sufficient for the full dashboard experience.

3. Ensure your Supabase database has the required tables:
- `profiles` - User profiles with subscription information
- `trips` - Trip records with mileage and duration data

### Running the Dashboard

#### Easy Method
```bash
./dashboard_mtp
```

#### Manual Method
```bash
source venv/bin/activate  # Activate virtual environment
streamlit run dashboard.py --server.address localhost
```

The dashboard will be available at:
- Local: http://localhost:8501

If you intentionally want LAN access, opt in explicitly:

```bash
streamlit run dashboard.py --server.address 0.0.0.0
```

## 📁 Project Structure

```
Dashboard-MTP/
├── dashboard.py           # Main Streamlit application
├── supabase_client.py     # Enhanced Supabase client with retry logic
├── database_queries.py    # Advanced SQL query functions
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (not in git)
├── .gitignore           # Git ignore rules
├── dashboard_mtp        # Quick start script
├── quick_setup.sh       # Automated setup script
├── setup.py             # Interactive setup wizard
└── README.md           # This file
```

## 🛠️ Technical Stack

- **Frontend Framework**: Streamlit 1.39.0
- **Database**: Supabase (PostgreSQL)
- **Data Processing**: Pandas 2.2.2
- **Visualizations**: Plotly 5.24.1
- **Environment Management**: python-dotenv
- **Additional Libraries**: NumPy, Streamlit-extras

## Security Notes

- `SUPABASE_KEY` must be kept server-side only. Do not expose it to client-side code or publish it in frontend assets.
- The current dashboard expects a service role key because it reads Supabase Auth Admin users.
- The launcher defaults to `localhost` to avoid unintentionally exposing the dashboard on the local network.

## 📊 Database Schema

### Profiles Table
```sql
CREATE TABLE profiles (
    id UUID PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE,
    full_name TEXT,
    phone_number TEXT,
    subscription_tier TEXT,
    -- Additional fields as needed
);
```

### Trips Table
```sql
CREATE TABLE trips (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES profiles(id),
    created_at TIMESTAMP WITH TIME ZONE,
    mileage NUMERIC,
    actual_distance NUMERIC,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    -- Additional fields as needed
);
```

## 🔧 Advanced Features

### Enhanced Supabase Client
- Automatic retry logic with exponential backoff
- Connection health monitoring
- Rate limiting protection
- Query metrics tracking
- Multi-environment support

### Performance Optimizations
- Streamlit caching for data queries
- Efficient data aggregation
- Optimized database queries
- Timezone-aware datetime handling

### Data Export
- CSV export for user analytics
- At-risk users reports
- Retention analysis exports

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Supabase](https://supabase.com/)
- Charts by [Plotly](https://plotly.com/)

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## 🔮 Future Enhancements

- [ ] Email/Slack notifications for key metrics
- [ ] PDF report generation
- [ ] Advanced predictive analytics
- [ ] A/B testing results tracking
- [ ] Mobile-responsive design improvements
- [ ] Dark mode toggle
- [ ] Multi-language support
- [ ] API endpoints for external integrations

---

**Built with ❤️ for Mileage Tracker Pro**
