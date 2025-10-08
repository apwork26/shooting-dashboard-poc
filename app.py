"""
🎯Shooting Analysis Dashboard - COMPLETE VERSION
=====================================================
Interactive dashboard with comprehensive analysis and visualizations

Features:
- 6 Main Tabs: Overview, Qualification, Finals, Athlete Deep Dive, Cross-Tournament, Insights
- Interactive filtering and drill-down capabilities
- Rich visualizations using Plotly
- Downloadable reports and data exports

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import parser and enhanced analytics engine
try:
    from data_parser import DataParser
    from analytics_engine import ShootingAnalyticsEngine
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please ensure data_parser.py and analytics_engine.py are in the same directory")
    st.stop()

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    page_title="🎯Shooting Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============ DATA LOADING FUNCTIONS ============
@st.cache_data
def load_parsed_data(csv_dir='data/processed'):
    """Load pre-parsed CSV data"""
    csv_path = Path(csv_dir)

    if not csv_path.exists():
        return None, None

    # Try to load qualification and finals data
    qual_files = list(csv_path.glob('*qualification*.csv'))
    finals_files = list(csv_path.glob('*finals*.csv'))

    if not qual_files:
        return None, None

    # Load all qualification files
    qual_dfs = []
    for file in qual_files:
        try:
            df = pd.read_csv(file)
            qual_dfs.append(df)
        except Exception as e:
            st.warning(f"Could not load {file.name}: {e}")

    # Load all finals files
    finals_dfs = []
    for file in finals_files:
        try:
            df = pd.read_csv(file)
            # Parse shots from string if needed
            if 'shots' in df.columns and df['shots'].dtype == 'object':
                df['shots'] = df['shots'].apply(lambda x: [float(s) for s in str(x).split(',') if s.strip()])
            finals_dfs.append(df)
        except Exception as e:
            st.warning(f"Could not load {file.name}: {e}")

    qual_df = pd.concat(qual_dfs, ignore_index=True) if qual_dfs else pd.DataFrame()
    finals_df = pd.concat(finals_dfs, ignore_index=True) if finals_dfs else pd.DataFrame()

    return qual_df, finals_df

@st.cache_data
def parse_pdfs(pdf_directory='data/raw'):
    """Parse PDFs if CSV data not available"""
    parser = DataParser(pdf_directory)
    qual_df, finals_df = parser.parse_all_pdfs()
    parser.export_results(qual_df, finals_df, 'data/processed')

    return qual_df, finals_df

# ============ MAIN APPLICATION ============
def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Shooting Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading data..."):
        qual_df, finals_df = load_parsed_data()

        if qual_df is None or qual_df.empty:
            st.warning("No processed data found. Attempting to parse PDFs...")
            qual_df, finals_df = parse_pdfs()

            if qual_df.empty:
                st.error("No data available. Please add PDF files to data/raw/ directory.")
                st.stop()

    # Initialize analytics engine
    engine = ShootingAnalyticsEngine(qual_df, finals_df)
    qual_metrics = engine.calculate_qualification_metrics()
    finals_metrics = engine.calculate_finals_metrics() if not finals_df.empty else pd.DataFrame()

    # ============ SIDEBAR FILTERS ============
    st.sidebar.title("🔍 Filters")

    # Tournament filter with safety check
    if 'full_tournament_name' in qual_metrics.columns:
        tournament_col = 'full_tournament_name'
        all_tournaments = sorted([str(t) for t in qual_metrics['full_tournament_name'].unique() if pd.notna(t)])
    else:
        tournament_col = 'tournament'
        all_tournaments = sorted([str(t) for t in qual_metrics['tournament'].unique() if pd.notna(t)])

    selected_tournaments = st.sidebar.multiselect(
        "Select Tournaments",
        options=all_tournaments,
        default=all_tournaments
    )

    # Gender filter
    gender_options = ['All'] + list(qual_metrics['gender'].unique())
    selected_gender = st.sidebar.selectbox("Gender", gender_options)

    # Country filter
    country_options = ['All'] + sorted([str(c) for c in qual_metrics['country'].unique() if pd.notna(c)])
    selected_country = st.sidebar.selectbox("Country", country_options)

    # Apply filters
    filtered_qual = qual_metrics[qual_metrics[tournament_col].isin(selected_tournaments)]
    if selected_gender != 'All':
        filtered_qual = filtered_qual[filtered_qual['gender'] == selected_gender]
    if selected_country != 'All':
        filtered_qual = filtered_qual[filtered_qual['country'] == selected_country]

    filtered_finals = finals_metrics
    if not finals_metrics.empty:
        finals_tournament_col = 'full_tournament_name' if 'full_tournament_name' in finals_metrics.columns else 'tournament'
        filtered_finals = finals_metrics[finals_metrics[finals_tournament_col].isin(selected_tournaments)]
        if selected_gender != 'All':
            filtered_finals = filtered_finals[filtered_finals['gender'] == selected_gender]
        if selected_country != 'All':
            filtered_finals = filtered_finals[filtered_finals['country'] == selected_country]

    # ============ MAIN TABS ============
    tabs = st.tabs(["📊 Overview", "🎯 Qualification", "🏆 Finals", "👤 Athlete Deep Dive", "🔄 Cross-Tournament", "💡 Insights"])

    # ============ TAB 1: OVERVIEW ============
    with tabs[0]:
        st.header("Dashboard Overview")

        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Athletes", len(filtered_qual))
        with col2:
            qualified = len(filtered_qual[filtered_qual['qualified']])
            st.metric("Qualified", qualified)
        with col3:
            avg_score = filtered_qual['total_score'].mean()
            st.metric("Avg Score", f"{avg_score:.1f}")
        with col4:
            if not filtered_finals.empty:
                st.metric("Finalists", len(filtered_finals))
            else:
                st.metric("Finalists", "N/A")
        with col5:
            tournaments_count = len(filtered_qual[tournament_col].unique())
            st.metric("Tournaments", tournaments_count)

        st.markdown("---")

        # Top performers
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏅 Top Qualification Scores")
            display_cols = ['athlete_name', 'country', tournament_col, 'total_score', 'rank']
            top_qual = filtered_qual.nlargest(10, 'total_score')[display_cols].copy()
            top_qual.columns = ['Athlete', 'Country', 'Tournament', 'Score', 'Rank']
            st.dataframe(top_qual, hide_index=True, use_container_width=True)

        with col2:
            st.subheader("🏆 Top Finals Performances")
            if not filtered_finals.empty:
                finals_display_cols = ['athlete_name', 'country', finals_tournament_col, 'total_score', 'rank']
                top_finals = filtered_finals.nlargest(10, 'total_score')[finals_display_cols].copy()
                top_finals.columns = ['Athlete', 'Country', 'Tournament', 'Score', 'Rank']
                st.dataframe(top_finals, hide_index=True, use_container_width=True)
            else:
                st.info("No finals data available")

        st.markdown("---")

        # Tournament comparison
        st.subheader("📍 Tournament Comparison")
        tournament_stats = filtered_qual.groupby(tournament_col).agg({
            'total_score': ['mean', 'max', 'min', 'std'],
            'athlete_name': 'count'
        }).round(2)
        tournament_stats.columns = ['Avg Score', 'Max Score', 'Min Score', 'Std Dev', 'Athletes']
        st.dataframe(tournament_stats, use_container_width=True)

    # ============ TAB 2: QUALIFICATION ANALYSIS ============
    with tabs[1]:
        st.header("Qualification Analysis")

        # Display full table
        st.subheader("Qualification Results")
        display_cols = ['athlete_name', 'country', tournament_col, 'total_score', 'rank', 'consistency_score', 'qualified']
        st.dataframe(filtered_qual[display_cols], hide_index=True, use_container_width=True)

        # Series consistency visualization
        if not filtered_qual.empty and len(filtered_qual) > 0:
            st.subheader("Series Consistency Heatmap")
            top_n = min(15, len(filtered_qual))
            top_athletes = filtered_qual.nlargest(top_n, 'total_score')

            series_data = []
            athlete_labels = []
            for _, athlete in top_athletes.iterrows():
                series_scores = [athlete[f'series_{i}'] for i in range(1, 7)]
                series_data.append(series_scores)
                athlete_labels.append(f"{athlete['athlete_name']} ({athlete['total_score']:.0f})")

            fig = go.Figure(data=go.Heatmap(
                z=series_data,
                x=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
                y=athlete_labels,
                colorscale='RdYlGn',
                text=series_data,
                texttemplate='%{text:.1f}',
                textfont={"size": 10}
            ))
            fig.update_layout(
                title=f'Series Performance Heatmap (Top {top_n} Athletes)',
                xaxis_title='Series',
                yaxis_title='Athlete (Total Score)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    # ============ TAB 3: FINALS ANALYSIS ============
    with tabs[2]:
        st.header("Finals Analysis")

        if filtered_finals.empty:
            st.info("No finals data available for selected filters")
        else:
            # Finals results table
            st.subheader("Finals Results")
            display_cols = ['athlete_name', 'country', finals_tournament_col, 'total_score', 'rank', 'shot_avg', 'gqs_percentage']
            st.dataframe(filtered_finals[display_cols], hide_index=True, use_container_width=True)

            # Shot quality visualization
            if len(filtered_finals) > 0:
                st.subheader("Shot Quality Distribution")
                top_finalists = filtered_finals.nlargest(8, 'total_score')

                data = []
                for _, athlete in top_finalists.iterrows():
                    data.append({
                        'Athlete': athlete['athlete_name'],
                        'Excellent (≥10.5)': athlete['excellent_percentage'],
                        'GQS (10.0-10.4)': athlete['gqs_percentage'] - athlete['excellent_percentage'],
                        'Average (9.5-9.9)': athlete['average_percentage'],
                        'Poor (<9.5)': athlete['poor_percentage']
                    })

                df_chart = pd.DataFrame(data)

                fig = go.Figure()
                colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
                categories = ['Excellent (≥10.5)', 'GQS (10.0-10.4)', 'Average (9.5-9.9)', 'Poor (<9.5)']

                for i, category in enumerate(categories):
                    fig.add_trace(go.Bar(
                        name=category,
                        x=df_chart['Athlete'],
                        y=df_chart[category],
                        marker_color=colors[i]
                    ))

                fig.update_layout(
                    barmode='stack',
                    title='Shot Quality Distribution (Top 8 Finalists)',
                    yaxis_title='Percentage (%)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

    # ============ TAB 4: ATHLETE DEEP DIVE ============
    with tabs[3]:
        st.header("Athlete Deep Dive")

        # Athlete selector
        all_athletes = sorted(filtered_qual['athlete_name'].unique())
        selected_athlete = st.selectbox("Select Athlete", all_athletes)

        if selected_athlete:
            athlete_qual = filtered_qual[filtered_qual['athlete_name'] == selected_athlete]

            # Performance summary
            st.subheader(f"Performance Summary - {selected_athlete}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_score = athlete_qual['total_score'].mean()
                st.metric("Avg Score", f"{avg_score:.1f}")
            with col2:
                best_score = athlete_qual['total_score'].max()
                st.metric("Best Score", f"{best_score:.0f}")
            with col3:
                appearances = len(athlete_qual)
                st.metric("Appearances", appearances)
            with col4:
                qualification_rate = (athlete_qual['qualified'].sum() / len(athlete_qual) * 100)
                st.metric("Qualification Rate", f"{qualification_rate:.1f}%")

            # Tournament history
            st.subheader(f"Tournament History")
            history_cols = [tournament_col, 'rank', 'total_score', 'consistency_score', 'qualified']
            # Filter columns that exist
            history_cols = [col for col in history_cols if col in athlete_qual.columns]
            history = athlete_qual[history_cols].sort_values(tournament_col)
            st.dataframe(history, hide_index=True, use_container_width=True)

            # Finals performance (if available)
            if not filtered_finals.empty:
                athlete_finals = filtered_finals[filtered_finals['athlete_name'] == selected_athlete]

                if not athlete_finals.empty:
                    st.subheader("Finals Performances")

                    # Tournament selector for shot sequence
                    tournament_col_finals = 'full_tournament_name' if 'full_tournament_name' in athlete_finals.columns else 'tournament'
                    tournament_select = st.selectbox(
                        "Select Tournament for Shot Details",
                        athlete_finals[tournament_col_finals].unique(),
                        key="athlete_tournament"
                    )

                    athlete_finals_detail = athlete_finals[athlete_finals[tournament_col_finals] == tournament_select].iloc[0]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final Score", f"{athlete_finals_detail['total_score']:.1f}")
                    with col2:
                        st.metric("Rank", int(athlete_finals_detail['rank']))
                    with col3:
                        st.metric("Shot Average", f"{athlete_finals_detail['shot_avg']:.2f}")

                    # Shot sequence
                    if 'shots_string' in athlete_finals_detail:
                        shots = [float(x) for x in str(athlete_finals_detail['shots_string']).split(',') if x.strip()]
                        shot_df = pd.DataFrame({
                            'Shot #': list(range(1, len(shots) + 1)),
                            'Score': shots
                        })

                        fig = px.line(shot_df, x='Shot #', y='Score', markers=True, 
                                    title=f'Shot Sequence - {tournament_select}')
                        fig.add_hline(y=10.0, line_dash="dash", line_color="green", 
                                    annotation_text="GQS Threshold")
                        fig.add_hline(y=10.5, line_dash="dash", line_color="blue", 
                                    annotation_text="Excellent Threshold")
                        st.plotly_chart(fig, use_container_width=True)

    # ============ TAB 5: CROSS-TOURNAMENT ANALYSIS ============
    with tabs[4]:
        st.header("Cross-Tournament Analysis")

        st.subheader("Athlete Performance Across Tournaments")

        # Select athletes for comparison
        top_athletes = filtered_qual.groupby('athlete_name')['total_score'].mean().nlargest(20).index.tolist()
        selected_for_comparison = st.multiselect(
            "Select athletes to compare (max 5)",
            top_athletes,
            default=top_athletes[:3] if len(top_athletes) >= 3 else top_athletes
        )

        if selected_for_comparison:
            data = []
            for athlete in selected_for_comparison:
                athlete_data = filtered_qual[filtered_qual['athlete_name'] == athlete]
                for _, row in athlete_data.iterrows():
                    data.append({
                        'Athlete': athlete,
                        'Tournament': row[tournament_col],
                        'Score': row['total_score'],
                        'Rank': row['rank']
                    })

            df_comparison = pd.DataFrame(data)

            fig = px.line(df_comparison, x='Tournament', y='Score', color='Athlete', 
                         markers=True, title='Performance Across Tournaments')
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            st.subheader("Performance Summary")
            summary = filtered_qual[filtered_qual['athlete_name'].isin(selected_for_comparison)].groupby('athlete_name').agg({
                'total_score': ['mean', 'max', 'min', 'std'],
                tournament_col: 'count',
                'qualified': 'sum'
            }).round(2)
            summary.columns = ['Avg Score', 'Max Score', 'Min Score', 'Std Dev', 'Tournaments', 'Qualifications']
            st.dataframe(summary, use_container_width=True)

    # ============ TAB 6: INSIGHTS ============
    with tabs[5]:
        st.header("Performance Insights")

        st.subheader("Key Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Qualification Statistics")
            st.metric("Average Score", f"{filtered_qual['total_score'].mean():.2f}")
            st.metric("Median Score", f"{filtered_qual['total_score'].median():.2f}")
            st.metric("Std Deviation", f"{filtered_qual['total_score'].std():.2f}")

        with col2:
            st.markdown("### Competition Statistics")
            st.metric("Total Competitions", len(filtered_qual[tournament_col].unique()))
            st.metric("Total Athletes", filtered_qual['athlete_name'].nunique())
            st.metric("Countries Represented", filtered_qual['country'].nunique())

        # Distribution plot
        st.subheader("Score Distribution")
        fig = px.histogram(filtered_qual, x='total_score', nbins=30, 
                          title='Distribution of Qualification Scores')
        fig.add_vline(x=filtered_qual['total_score'].mean(), line_dash="dash", 
                     line_color="red", annotation_text="Mean")
        st.plotly_chart(fig, use_container_width=True)

        # Top countries
        st.subheader("Performance by Country")
        country_stats = filtered_qual.groupby('country').agg({
            'total_score': 'mean',
            'athlete_name': 'count',
            'qualified': 'sum'
        }).round(2)
        country_stats.columns = ['Avg Score', 'Athletes', 'Qualifications']
        country_stats = country_stats.sort_values('Avg Score', ascending=False).head(10)
        st.dataframe(country_stats, use_container_width=True)

if __name__ == "__main__":
    main()
