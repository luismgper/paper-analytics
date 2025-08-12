import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl
import numpy as np
import io
import ollama
from matplotlib import patheffects
import json
from typing import Optional

# Your specific imports and initialization
import src.papers.domain.multimodal_paper_query as mpq
from src.papers.domain.multimodal_paper_query import Conference
from src.papers.io.db import Milvus, Neo4j, SQLite
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import your analytics class (adjust path as needed)
from src.papers.domain.paper_analytics import PaperAnalytics
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

class StreamlitPaperAnalytics:
    """Streamlit interface for Paper Analytics using matplotlib"""
    PROMPTS = {
        "paper": """containing counts of papers published,categorized by continent, conference, and year.""",
        "citation": "containing counts of paper citations from papers published in conferences,categorized by continent, conference, and year.",
        "paper_and_citation": "containing counts of papers published in conferences and their citations,categorized by continent, conference, and year.",
        "committee": "containing counts of committee members of conferences,categorized by continent, conference, and year.",
    }
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.setup_matplotlib_style()
        # Automatically initialize database connection
        self.auto_initialize_connection()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Paper Analytics Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def setup_matplotlib_style(self):
        """Configure matplotlib and seaborn styling"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'analytics_client' not in st.session_state:
            st.session_state.analytics_client = None
        if 'query_results' not in st.session_state:
            st.session_state.query_results = {}
        if 'connection_status' not in st.session_state:
            st.session_state.connection_status = None
        if 'connection_error' not in st.session_state:
            st.session_state.connection_error = None
            
    def auto_initialize_connection(self):
        """Automatically initialize database connection on app startup"""
        if st.session_state.analytics_client is None:
            try:
                # Environment variables
                MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")
                MILVUS_ALIAS = os.getenv("MILVUS_ALIAS") 
                MILVUS_HOST = os.getenv("MILVUS_HOST")
                MILVUS_PORT = os.getenv("MILVUS_PORT")
                NEO4J_URI = os.getenv("NEO4J_URI")
                NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
                NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
                NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
                
                # Check if all required environment variables are set
                required_vars = {
                    "MILVUS_COLLECTION": MILVUS_COLLECTION,
                    "MILVUS_ALIAS": MILVUS_ALIAS,
                    "MILVUS_HOST": MILVUS_HOST,
                    "MILVUS_PORT": MILVUS_PORT,
                    "NEO4J_URI": NEO4J_URI,
                    "NEO4J_USERNAME": NEO4J_USERNAME,
                    "NEO4J_PASSWORD": NEO4J_PASSWORD,
                    "NEO4J_DATABASE": NEO4J_DATABASE
                }
                
                missing_vars = [var for var, value in required_vars.items() if not value]
                
                if missing_vars:
                    st.session_state.connection_status = "missing_vars"
                    st.session_state.connection_error = f"Missing environment variables: {', '.join(missing_vars)}"
                    return
                
                # Initialize clients
                milvus_client = Milvus(
                    collection=MILVUS_COLLECTION,
                    alias=MILVUS_ALIAS,
                    host=MILVUS_HOST,
                    port=int(MILVUS_PORT),
                )
                
                neo4j_client = Neo4j(
                    uri=NEO4J_URI,
                    username=NEO4J_USERNAME,
                    password=NEO4J_PASSWORD,
                    database=NEO4J_DATABASE
                )
                
                query_client = mpq.MultiModalPaperQuery(
                    relational_db_client=SQLite, 
                    vector_db_client=milvus_client, 
                    graph_db_client=neo4j_client
                )
                
                st.session_state.analytics_client = PaperAnalytics(query_client)
                st.session_state.connection_status = "connected"
                st.session_state.connection_error = None
                
            except Exception as e:
                st.session_state.connection_status = "error"
                st.session_state.connection_error = str(e)
                st.session_state.analytics_client = None
    
    def setup_connection(self):
        """Display connection status and provide manual override if needed"""
        st.sidebar.header("🔗 Database Connection")
        
        # Display current connection status
        if st.session_state.connection_status == "connected":
            st.sidebar.success("✅ Database Connected Successfully!")

                
        elif st.session_state.connection_status == "missing_vars":
            st.sidebar.error("❌ Missing Environment Variables")
            st.sidebar.error(st.session_state.connection_error)
                           
        elif st.session_state.connection_status == "error":
            st.sidebar.error("❌ Connection Failed")
            st.sidebar.error(st.session_state.connection_error)
            
            # Retry button
            if st.sidebar.button("🔄 Retry Connection"):
                st.session_state.analytics_client = None
                st.session_state.connection_status = None
                st.session_state.connection_error = None
                self.auto_initialize_connection()
                st.rerun()
                
        else:
            st.sidebar.warning("⚠️ Connection Status Unknown")
                
    def render_filters_sidebar(self, analysis_type=None):
        """Render filter controls in sidebar based on selected analysis type"""
        st.sidebar.header("🔍 Filters")
        # Define which filters are needed for each analysis type
        filter_requirements = {
            "Papers by Conference, Continent and Year": ["text","conferences", "years", "continents"],
            # "Papers by Conference and Continent": ["text", "conferences", "continents"],
            "Citations by Conference, Continent and Year": ["text","conferences", "years", "continents", "cited_continents"],
            # "Citations by Conference and Continent": ["text", "conferences", "continents", "cited_continents"],            
            "Citations by Conference and Source and Cited Continent": ["text", "conferences", "continents", "cited_continents"],            
            "Committees by Conference, Country and Year": ["conferences", "years"],
            "Committees by Continent and Year": ["conferences", "continents", "years"],
            "Committees by Continent": ["conferences", "continents"]
        }
        
        filters = {}
        
        if analysis_type and analysis_type in filter_requirements:
            required_filters = filter_requirements[analysis_type]
            # st.sidebar.info(f"Filters for: **{analysis_type}**")
        else:
            required_filters = ["conferences", "years", "continents"]  # Default: show all
            if not analysis_type:
                st.sidebar.info("Select an analysis type to see relevant filters")
        
        # Conference filter - using Conference enum
        if "text" in required_filters:
            text = st.sidebar.text_input(
                "Search papers by title and content topic",
                placeholder="Enter keywords to search papers...",
                key="text_query_input",
                help="Search for papers containing specific keywords in title or content"
            )
            filters['text'] = text if text else None
        else:
            filters['text'] = None    
        
        # Conference filter - using Conference enum
        if "conferences" in required_filters:
            available_conferences = [conf.value for conf in Conference]
            conferences = st.sidebar.multiselect(
                "Select conferences",
                options=available_conferences,
                key="selected_conferences",
                help="Filter by specific conferences"
            )
            # Convert back to Conference enum objects
            filters['conferences'] = [Conference(conf) for conf in conferences] if conferences else None
        else:
            filters['conferences'] = None
        
        # Year filter
        if "years" in required_filters:
            years = st.sidebar.multiselect(
                "Select years",
                options=[str(year) for year in range(2012, 2024)],
                key="selected_years",
                help="Filter by specific years"
            )
            filters['years'] = years if years else None
        else:
            filters['years'] = None
        
        # Continent filter
        if "continents" in required_filters:
            continents = st.sidebar.multiselect(
                "Select continents",
                options=["Asia", "Europe", "North America", "South America", "Africa", "Oceania"],
                key="selected_continents",
                help="Filter by specific continents"
            )
            filters['continents'] = continents if continents else None
        else:
            filters['continents'] = None
            
        # Cited continent filter
        if "cited_continents" in required_filters:
            cited_continents = st.sidebar.multiselect(
                "Select cited continents",
                options=["Asia", "Europe", "North America", "South America", "Africa", "Oceania"],
                key="selected_cited_continents",
                help="Filter by specific cited continents"
            )
            filters['cited_continents'] = cited_continents if cited_continents else None
        else:
            filters['cited_continents'] = None            
        
        ai_analysis = st.sidebar.checkbox("Use AI to explain results")
        
        # Show a summary of active filters
        active_filters = [k for k, v in filters.items() if v is not None and len(v) > 0]
        if active_filters:
            st.sidebar.success(f"✅ Active filters: {', '.join(active_filters)}")
        else:
            st.sidebar.info("ℹ️ No filters applied (showing all data)")
        
        return filters, ai_analysis
    
    def display_dataframe_with_download(self, df: pl.DataFrame, title: str, key: str):
        """Display DataFrame with download option"""
        st.subheader(f"📋 {title}")
        
        if df.height > 0:
            # Convert to pandas for better display
            df_pandas = df.to_pandas()
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", df.height)
            with col2:
                if 'paper_count' in df.columns:
                    st.metric("Total Papers", df.select(pl.sum('paper_count')).item())
                elif 'committee_count' in df.columns:
                    st.metric("Total Committees", df.select(pl.sum('committee_count')).item())
            
            # Display table
            st.dataframe(df_pandas, use_container_width=True, key=f"df_{key}")
            
            # # Download button
            # csv = df_pandas.to_csv(index=False)
            # st.download_button(
            #     label=f"📥 Download {title} as CSV",
            #     data=csv,
            #     file_name=f"{title.lower().replace(' ', '_')}.csv",
            #     mime="text/csv",
            #     key=f"download_{key}"
            # )
        else:
            st.warning("No data found with current filters.")
    
    def create_matplotlib_chart(self, fig, title="Chart"):
        """Helper to display matplotlib figure in streamlit"""
        st.pyplot(fig)
        
        # Option to download the plot
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label=f"📥 Download {title} as PNG",
            data=buf.getvalue(),
            file_name=f"{title.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
        plt.close(fig)

    def create_paper_visualizations(self, df: pl.DataFrame, do_ai_analysis: bool):
        """Create plotly visualizations for paper data"""
        if df.height == 0:
            return
        
        if do_ai_analysis:
            self.print_llm_analysis(df, "paper")        
            
        df_pandas = df.to_pandas()
        
        # Create tabs for different visualizations
        if 'source_year' in df_pandas.columns:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Bar Chart (totals)", "📊 Bar Chart (percentages)", "📈 Line Chart (totals)", "📈 Line Chart (percentages)", "🥧 Pie Chart"])
        else:
            tab1, tab2, tab5 = st.tabs(["📊 Bar Chart (totals)", "📊 Bar Chart (percentages)", "🥧 Pie Chart"])
                
        with tab1:
            if 'source_conference' in df_pandas.columns:
                grouped_data = df_pandas.groupby(['source_conference', 'source_predominant_continent'])['paper_count'].sum().reset_index()
                
                # Create grouped bar chart
                conferences = grouped_data['source_conference'].unique()
                continents = grouped_data['source_predominant_continent'].unique()
                
                fig = go.Figure()
                
                # Add bars for each continent
                for continent in continents:
                    continent_data = grouped_data[grouped_data['source_predominant_continent'] == continent]
                    values = [continent_data[continent_data['source_conference'] == conf]['paper_count'].sum() 
                            if conf in continent_data['source_conference'].values else 0 
                            for conf in conferences]
                    
                    fig.add_trace(go.Bar(
                        name=continent,
                        x=conferences,
                        y=values,
                        opacity=0.8
                    ))
                
                # Update layout
                fig.update_layout(
                    title='Papers by conference and continent',
                    xaxis_title='Conference',
                    yaxis_title='Number of papers',
                    barmode='group',
                    height=500,
                    xaxis=dict(
                        tickangle=45,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)'
                    ),
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                    
        with tab2:
            if 'source_conference' in df_pandas.columns:
                df_pandas_percentage = (
                    df
                    .group_by("source_conference", "source_predominant_continent")
                    .agg([
                        pl.sum("paper_count")
                    ])                    
                    .with_columns([
                        (
                            pl.col('paper_count') / 
                            pl.col('paper_count').sum().over(['source_conference']) * 100
                        )
                        .round(2)
                        .alias('paper_percentage')
                    ])
                    .sort(['source_conference', 'source_predominant_continent'])                    
                ).to_pandas()
                print(df_pandas_percentage)
                grouped_data = df_pandas_percentage
                
                # Create grouped bar chart
                conferences = grouped_data['source_conference'].unique()
                continents = grouped_data['source_predominant_continent'].unique()
                
                fig = go.Figure()
                
                # Add bars for each continent
                for continent in continents:
                    continent_data = grouped_data[grouped_data['source_predominant_continent'] == continent]
                    values = [continent_data[continent_data['source_conference'] == conf]['paper_percentage'].sum() 
                            if conf in continent_data['source_conference'].values else 0 
                            for conf in conferences]
                    
                    fig.add_trace(go.Bar(
                        name=continent,
                        x=conferences,
                        y=values,
                        opacity=0.8
                    ))
                
                # Update layout
                fig.update_layout(
                    title='Papers by conference and continent (percentages)',
                    xaxis_title='Conference',
                    yaxis_title='Percentage of papers (%)',
                    barmode='group',
                    height=500,
                    xaxis=dict(
                        tickangle=45,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)'
                    ),
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.01
                    )
                )
                
                print("ok")
                st.plotly_chart(fig, use_container_width=True)
        
        if 'source_year' in df_pandas.columns:
            with tab3:               
                # Get unique conferences
                conferences = df_pandas['source_conference'].unique()
                num_conferences = len(conferences)
                
                # Calculate subplot grid dimensions
                cols = min(2, num_conferences)
                rows = (num_conferences + cols - 1) // cols
                
                # Create subplot titles
                subplot_titles = [f'Paper trends - {conf}' for conf in conferences]
                
                # Create subplots
                fig = make_subplots(
                    rows=rows, 
                    cols=cols,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.08,
                    horizontal_spacing=0.1
                )
                
                # Get all unique continents for consistent coloring
                all_continents = df_pandas['source_predominant_continent'].unique()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                continent_colors = {continent: colors[i % len(colors)] for i, continent in enumerate(all_continents)}
                
                # Plot each conference
                for i, conference in enumerate(conferences):
                    row = i // cols + 1
                    col = i % cols + 1
                    
                    # Filter data for this conference
                    conference_data = df_pandas[df_pandas['source_conference'] == conference]
                    yearly_data = conference_data.groupby(['source_year', 'source_predominant_continent'])['paper_count'].sum().reset_index()
                    
                    # Plot each continent for this conference
                    for continent in yearly_data['source_predominant_continent'].unique():
                        continent_data = yearly_data[yearly_data['source_predominant_continent'] == continent]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=continent_data['source_year'],
                                y=continent_data['paper_count'],
                                mode='lines+markers',
                                name=continent,
                                line=dict(width=2, color=continent_colors[continent]),
                                marker=dict(size=6, color=continent_colors[continent]),
                                legendgroup=continent,
                                showlegend=(i == 0),
                            ),
                            row=row, col=col
                        )
                
                # Update layout
                fig.update_layout(
                    height=600 * rows,
                    title_text="Paper trends over time",
                    title_x=0.5,
                    showlegend=True
                )
                
                # Update axes
                fig.update_xaxes(
                    title_text="Year",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)',
                    tickangle=45
                )
                
                fig.update_yaxes(
                    title_text="Number of papers",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with tab4:               
                # Get unique conferences
                conferences = df_pandas['source_conference'].unique()
                num_conferences = len(conferences)
                
                df_pandas_percentage = (
                    df                  
                    .with_columns([
                        (
                            pl.col('paper_count') / 
                            pl.col('paper_count').sum().over(['source_conference', 'source_year']) * 100
                        )
                        .round(2)
                        .alias('paper_percentage')
                    ])
                    .sort(['source_conference', 'source_predominant_continent'])                    
                ).to_pandas()
                print(df_pandas_percentage)                
                
                # Calculate subplot grid dimensions
                cols = min(2, num_conferences)
                rows = (num_conferences + cols - 1) // cols
                
                # Create subplot titles
                subplot_titles = [f'Paper trends - {conf}' for conf in conferences]
                
                # Create subplots
                fig = make_subplots(
                    rows=rows, 
                    cols=cols,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.08,
                    horizontal_spacing=0.1
                )
                
                # Get all unique continents for consistent coloring
                all_continents = df_pandas_percentage['source_predominant_continent'].unique()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                continent_colors = {continent: colors[i % len(colors)] for i, continent in enumerate(all_continents)}
                
                # Plot each conference
                for i, conference in enumerate(conferences):
                    row = i // cols + 1
                    col = i % cols + 1
                    
                    # Filter data for this conference
                    conference_data = df_pandas_percentage[df_pandas_percentage['source_conference'] == conference]
                    yearly_data = conference_data.groupby(['source_year', 'source_predominant_continent'])['paper_percentage'].sum().reset_index()
                    
                    # Plot each continent for this conference
                    for continent in yearly_data['source_predominant_continent'].unique():
                        continent_data = yearly_data[yearly_data['source_predominant_continent'] == continent]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=continent_data['source_year'],
                                y=continent_data['paper_percentage'],
                                mode='lines+markers',
                                name=continent,
                                line=dict(width=2, color=continent_colors[continent]),
                                marker=dict(size=6, color=continent_colors[continent]),
                                legendgroup=continent,
                                showlegend=(i == 0),
                            ),
                            row=row, col=col
                        )
                
                # Update layout
                fig.update_layout(
                    height=600 * rows,
                    title_text="Paper trends over time (percentages)",
                    title_x=0.5,
                    showlegend=True
                )
                
                # Update axes
                fig.update_xaxes(
                    title_text="Year",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)',
                    tickangle=45
                )
                
                fig.update_yaxes(
                    title_text="Percentage of papers (%)",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            if 'source_conference' in df_pandas.columns:
                conferences = df_pandas['source_conference'].unique()
                
                # Calculate number of rows and columns for subplots
                n_conferences = len(conferences)
                n_cols = min(3, n_conferences)
                n_rows = (n_conferences + n_cols - 1) // n_cols
                
                # Create subplot titles
                subplot_titles = []
                for conference in conferences:
                    conf_data = df_pandas[df_pandas['source_conference'] == conference]
                    total_papers = conf_data['paper_count'].sum()
                    subplot_titles.append(f'{conference}<br>({total_papers} papers)')
                
                # Create subplots with pie chart specs
                VERTICAL_SPACING = 0#.1#0.15
                HORIZONTAL_SPACING = 0.1#.05
                fig = make_subplots(
                    rows=n_rows, 
                    cols=n_cols,
                    subplot_titles=subplot_titles,
                    specs=[[{"type": "pie"}] * n_cols for _ in range(n_rows)],
                    vertical_spacing=VERTICAL_SPACING,  # Increased spacing to accommodate legends
                    horizontal_spacing=HORIZONTAL_SPACING   # Increased spacing to accommodate legends
                )
                
                # Define colors (equivalent to plt.cm.Set3)
                colors = [
                    '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
                    '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'
                ]
                
                # Plot each conference
                for i, conference in enumerate(conferences):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    
                    # Filter data for this conference
                    conf_data = df_pandas[df_pandas['source_conference'] == conference]
                    continent_totals = conf_data.groupby('source_predominant_continent')['paper_count'].sum()
                    
                    if len(continent_totals) > 0:
                        # Calculate percentages
                        total = continent_totals.sum()
                        percentages = [(value/total)*100 for value in continent_totals.values]
                        
                        # Create custom legend labels with percentages
                        legend_labels = [f'{continent}: {percent:.1f}%' 
                                        for continent, percent in zip(continent_totals.index, percentages)]
                        
                        # Select colors for this conference's continents
                        conf_colors = colors[:len(continent_totals)]
                        
                        fig.add_trace(
                            go.Pie(
                                labels=legend_labels,
                                values=continent_totals.values,
                                name=conference,
                                marker=dict(colors=conf_colors),
                                hovertemplate='%{label}<br>%{value} papers<br>%{percent}<extra></extra>',
                                textinfo='none',
                                showlegend=True,
                                legendgroup=f'group{i}',
                                legend=f'legend{i+1}' if i > 0 else 'legend',  # Assign to specific legend
                                domain=dict(row=row-1, column=col-1)
                            ),
                            row=row, col=col
                        )
                    else:
                        # For conferences with no data, add empty pie with annotation
                        fig.add_trace(
                            go.Pie(
                                labels=['No data'],
                                values=[1],
                                marker=dict(colors=['lightgray']),
                                showlegend=False,
                                textinfo='none',
                                hoverinfo='skip'
                            ),
                            row=row, col=col
                        )
                        
                        # Add annotation for "No data"
                        fig.add_annotation(
                            text="No data",
                            x=0.5, y=0.5,
                            xref=f"x{i+1 if i > 0 else ''}", 
                            yref=f"y{i+1 if i > 0 else ''}",
                            showarrow=False,
                            font=dict(size=12, color="black"),
                            align="center"
                        )
                        
                # Create legend configurations for each subplot
                legend_configs = {}
                for i, conference in enumerate(conferences):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    legend_x, legend_y = self.get_legend_position(row, col, n_rows, n_cols)
                    
                    legend_key = f'legend{i+1}' if i > 0 else 'legend'
                    legend_configs[legend_key] = dict(
                        x=legend_x,
                        y=legend_y,
                        xanchor="left",
                        # yanchor="middle",
                        yanchor="top",
                        font=dict(size=9),
                        bgcolor="rgba(255,255,255,0.5)",
                        bordercolor="rgba(0,0,0,0.1)",
                        borderwidth=1
                    )                        
                
                # Update layout
                fig.update_layout(
                    title=dict(
                        text='Distribution of papers by continent per conference',
                        x=0.5,
                        font=dict(size=16, color="black")
                    ),
                    height=600 * n_rows,
                    width=900 * n_cols, 
                    # showlegend=True,
                    # legend=dict(
                    #     orientation="v",
                    #     yanchor="middle",
                    #     y=0.5,
                    #     xanchor="left",
                    #     x=1.02,
                    #     font=dict(size=10)
                    # ),
                    # margin=dict(r=200),
                    **legend_configs  # Add all legend configurations
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
    def create_citation_visualizations(self, df: pl.DataFrame, do_ai_analysis: bool):
        """Create plotly visualizations for paper data"""
        if df.height == 0:
            return

        if do_ai_analysis:
            self.print_llm_analysis(df, "citation")    
            
        df_pandas = df.to_pandas()
        
        # Create tabs for different visualizations
        if 'source_year' in df_pandas.columns:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Bar Chart (totals)", "📊 Bar Chart (percentages)", "📈 Line Chart (totals)", "📈 Line Chart (percentages)", "🥧 Pie Chart"])
        else:
            tab1, tab5 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart"])
                
        with tab1:
            print(df_pandas.columns)
            if 'source_conference' in df_pandas.columns:
                grouped_data = df_pandas.groupby(['source_conference', 'cited_predominant_continent'])['paper_count'].sum().reset_index()
                
                # Create grouped bar chart
                conferences = grouped_data['source_conference'].unique()
                continents = grouped_data['cited_predominant_continent'].unique()
                
                fig = go.Figure()
                
                # Add bars for each continent
                for continent in continents:
                    continent_data = grouped_data[grouped_data['cited_predominant_continent'] == continent]
                    values = [continent_data[continent_data['source_conference'] == conf]['paper_count'].sum() 
                            if conf in continent_data['source_conference'].values else 0 
                            for conf in conferences]
                    
                    fig.add_trace(go.Bar(
                        name=continent,
                        x=conferences,
                        y=values,
                        opacity=0.8
                    ))
                
                # Update layout
                fig.update_layout(
                    title='Citations by conference and continent',
                    xaxis_title='Conference',
                    yaxis_title='Number of citations',
                    barmode='group',
                    width=1000,
                    height=571,
                    xaxis=dict(
                        tickangle=45,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)'
                    ),
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        if 'source_year' in df_pandas.columns:
            with tab2:
                print(df_pandas.columns)
                if 'source_conference' in df_pandas.columns:
                    df_pandas_percentage = (
                        df
                        .group_by("source_conference", "cited_predominant_continent")
                        .agg([
                            pl.sum("paper_count")
                        ])                    
                        .with_columns([
                            (
                                pl.col('paper_count') / 
                                pl.col('paper_count').sum().over(['source_conference']) * 100
                            )
                            .round(2)
                            .alias('paper_percentage')
                        ])
                        .sort(['source_conference', 'cited_predominant_continent'])                    
                    ).to_pandas()
                    print(df_pandas_percentage)
                    
                    grouped_data = df_pandas_percentage
                    
                    # Create grouped bar chart
                    conferences = grouped_data['source_conference'].unique()
                    continents = grouped_data['cited_predominant_continent'].unique()
                    
                    fig = go.Figure()
                    
                    # Add bars for each continent
                    for continent in continents:
                        continent_data = grouped_data[grouped_data['cited_predominant_continent'] == continent]
                        values = [continent_data[continent_data['source_conference'] == conf]['paper_percentage'].sum() 
                                if conf in continent_data['source_conference'].values else 0 
                                for conf in conferences]
                        
                        fig.add_trace(go.Bar(
                            name=continent,
                            x=conferences,
                            y=values,
                            opacity=0.8
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Citations by conference and continent (percentages)',
                        xaxis_title='Conference',
                        yaxis_title='Percentage of citations (%)',
                        barmode='group',
                        width=1000,
                        height=571,
                        xaxis=dict(
                            tickangle=45,
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128, 128, 128, 0.3)'
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128, 128, 128, 0.3)'
                        ),
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.01
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        if 'source_year' in df_pandas.columns:
            with tab3:                
                # Get unique conferences
                conferences = df_pandas['source_conference'].unique()
                num_conferences = len(conferences)
                
                # Calculate subplot grid dimensions
                cols = min(2, num_conferences)
                rows = (num_conferences + cols - 1) // cols
                
                # Create subplot titles
                subplot_titles = [f'Citation trends - {conf}' for conf in conferences]
                
                # Create subplots
                fig = make_subplots(
                    rows=rows, 
                    cols=cols,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.08,
                    horizontal_spacing=0.1
                )
                
                # Get all unique continents for consistent coloring
                all_continents = df_pandas['cited_predominant_continent'].unique()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                continent_colors = {continent: colors[i % len(colors)] for i, continent in enumerate(all_continents)}
                
                # Plot each conference
                for i, conference in enumerate(conferences):
                    row = i // cols + 1
                    col = i % cols + 1
                    
                    # Filter data for this conference
                    conference_data = df_pandas[df_pandas['source_conference'] == conference]
                    yearly_data = conference_data.groupby(['source_year', 'cited_predominant_continent'])['paper_count'].sum().reset_index()
                    
                    # Plot each continent for this conference
                    for continent in yearly_data['cited_predominant_continent'].unique():
                        continent_data = yearly_data[yearly_data['cited_predominant_continent'] == continent]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=continent_data['source_year'],
                                y=continent_data['paper_count'],
                                mode='lines+markers',
                                name=continent,
                                line=dict(width=2, color=continent_colors[continent]),
                                marker=dict(size=6, color=continent_colors[continent]),
                                legendgroup=continent,
                                showlegend=(i == 0),
                            ),
                            row=row, col=col
                        )
                
                # Update layout
                fig.update_layout(
                    height=600 * rows,
                    width=1000,
                    title_text="Citation trends over time (totals)",
                    title_x=0.5,
                    showlegend=True
                )
                
                # Update axes
                fig.update_xaxes(
                    title_text="Year",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)',
                    tickangle=45
                )
                
                fig.update_yaxes(
                    title_text="Number of citations",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        if 'source_year' in df_pandas.columns:
            with tab4:                
                # Get unique conferences
                conferences = df_pandas['source_conference'].unique()
                num_conferences = len(conferences)
                
                df_pandas_percentage = (
                    df                  
                    .with_columns([
                        (
                            pl.col('paper_count') / 
                            pl.col('paper_count').sum().over(['source_conference', 'source_year']) * 100
                        )
                        .round(2)
                        .alias('paper_percentage')
                    ])
                    .sort(['source_conference', 'cited_predominant_continent'])                    
                ).to_pandas()
                print(df_pandas_percentage)                    
                
                # Calculate subplot grid dimensions
                cols = min(2, num_conferences)
                rows = (num_conferences + cols - 1) // cols
                
                # Create subplot titles
                subplot_titles = [f'Citation trends - {conf}' for conf in conferences]
                
                # Create subplots
                fig = make_subplots(
                    rows=rows, 
                    cols=cols,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.08,
                    horizontal_spacing=0.1
                )
                
                # Get all unique continents for consistent coloring
                all_continents = df_pandas_percentage['cited_predominant_continent'].unique()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                continent_colors = {continent: colors[i % len(colors)] for i, continent in enumerate(all_continents)}
                
                # Plot each conference
                for i, conference in enumerate(conferences):
                    row = i // cols + 1
                    col = i % cols + 1
                    
                    # Filter data for this conference
                    conference_data = df_pandas_percentage[df_pandas_percentage['source_conference'] == conference]
                    yearly_data = conference_data.groupby(['source_year', 'cited_predominant_continent'])['paper_percentage'].sum().reset_index()
                    
                    # Plot each continent for this conference
                    for continent in yearly_data['cited_predominant_continent'].unique():
                        continent_data = yearly_data[yearly_data['cited_predominant_continent'] == continent]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=continent_data['source_year'],
                                y=continent_data['paper_percentage'],
                                mode='lines+markers',
                                name=continent,
                                line=dict(width=2, color=continent_colors[continent]),
                                marker=dict(size=6, color=continent_colors[continent]),
                                legendgroup=continent,
                                showlegend=(i == 0),
                            ),
                            row=row, col=col
                        )
                
                # Update layout
                fig.update_layout(
                    height=600 * rows,
                    width=1000,
                    title_text="Citation trends over time (percentages)",
                    title_x=0.5,
                    showlegend=True
                )
                
                # Update axes
                fig.update_xaxes(
                    title_text="Year",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)',
                    tickangle=45
                )
                
                fig.update_yaxes(
                    title_text="Percentage of citations (%)",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            if 'source_conference' in df_pandas.columns:
                conferences = df_pandas['source_conference'].unique()
                
                # Calculate number of rows and columns for subplots
                n_conferences = len(conferences)
                n_cols = min(3, n_conferences)
                n_rows = (n_conferences + n_cols - 1) // n_cols
                
                # Create subplot titles
                subplot_titles = []
                for conference in conferences:
                    conf_data = df_pandas[df_pandas['source_conference'] == conference]
                    total_papers = conf_data['paper_count'].sum()
                    subplot_titles.append(f'{conference}<br>({total_papers} papers)')
                
                # Create subplots with pie chart specs
                VERTICAL_SPACING = 0
                HORIZONTAL_SPACING = 0.1                
                fig = make_subplots(
                    rows=n_rows, 
                    cols=n_cols,
                    subplot_titles=subplot_titles,
                    specs=[[{"type": "pie"}] * n_cols for _ in range(n_rows)],
                    vertical_spacing=VERTICAL_SPACING, 
                    horizontal_spacing=HORIZONTAL_SPACING   
                )
                
                # Define colors (equivalent to plt.cm.Set3)
                colors = [
                    '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
                    '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'
                ]
                
                # Plot each conference
                for i, conference in enumerate(conferences):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    
                    # Filter data for this conference
                    conf_data = df_pandas[df_pandas['source_conference'] == conference]
                    continent_totals = conf_data.groupby('cited_predominant_continent')['paper_count'].sum()
                    
                    if len(continent_totals) > 0:
                        # Calculate percentages
                        total = continent_totals.sum()
                        percentages = [(value/total)*100 for value in continent_totals.values]
                        
                        # Create custom legend labels with percentages
                        legend_labels = [f'{continent}: {percent:.1f}%' 
                                        for continent, percent in zip(continent_totals.index, percentages)]
                        
                        # Select colors for this conference's continents
                        conf_colors = colors[:len(continent_totals)]
                        
                        fig.add_trace(
                            go.Pie(
                                labels=legend_labels,
                                values=continent_totals.values,
                                name=conference,
                                marker=dict(colors=conf_colors),
                                hovertemplate='%{label}<br>%{value} papers<br>%{percent}<extra></extra>',
                                textinfo='none',
                                showlegend=True,
                                legendgroup=f'group{i}',
                                legend=f'legend{i+1}' if i > 0 else 'legend',  # Assign to specific legend
                                domain=dict(row=row-1, column=col-1)
                            ),
                            row=row, col=col
                        )
                    else:
                        # For conferences with no data, add empty pie with annotation
                        fig.add_trace(
                            go.Pie(
                                labels=['No data'],
                                values=[1],
                                marker=dict(colors=['lightgray']),
                                showlegend=False,
                                textinfo='none',
                                hoverinfo='skip'
                            ),
                            row=row, col=col
                        )
                        
                        # Add annotation for "No data"
                        fig.add_annotation(
                            text="No data",
                            x=0.5, y=0.5,
                            xref=f"x{i+1 if i > 0 else ''}", 
                            yref=f"y{i+1 if i > 0 else ''}",
                            showarrow=False,
                            font=dict(size=12, color="black"),
                            align="center"
                        )

                # Create legend configurations for each subplot
                legend_configs = {}
                for i, conference in enumerate(conferences):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    legend_x, legend_y = self.get_legend_position(row, col, n_rows, n_cols)
                    
                    legend_key = f'legend{i+1}' if i > 0 else 'legend'
                    legend_configs[legend_key] = dict(
                        x=legend_x,
                        y=legend_y,
                        xanchor="left",
                        # yanchor="middle",
                        yanchor="top",
                        font=dict(size=9),
                        bgcolor="rgba(255,255,255,0.5)",
                        bordercolor="rgba(0,0,0,0.1)",
                        borderwidth=1
                    )                

                # Update layout
                fig.update_layout(
                    title=dict(
                        text='Distribution of citations by continent per conference',
                        x=0.5,
                        font=dict(size=16, color="black")
                    ),
                    height=600 * n_rows,  # Equivalent to 6 * n_rows
                    width=900 * n_cols,   # Increased width to accommodate individual legends
                    # margin=dict(l=50, r=100, t=80, b=50),  # Adjust margins
                    **legend_configs  # Add all legend configurations
                )
                
                st.plotly_chart(fig, use_container_width=True)            
                        
    def create_citation_by_source_and_cited_continent_visualizations(self, df: pl.DataFrame, do_ai_analysis: bool):
        """Create Sankey diagram visualizations for citation data"""
        if df.height == 0:
            return
        
        if do_ai_analysis:
            self.print_llm_analysis(df, "paper_and_citation")        
            
        df_pandas = df.to_pandas()
        
        # Create single tab for Sankey diagram
        tab1, tab2 = st.tabs(["🔄 Citation Flow Sankey (totals)", "🔄 Citation Flow Sankey (percentages)"])
        
        with tab1:
            if 'source_conference' in df_pandas.columns and 'source_predominant_continent' in df_pandas.columns:
                # Get unique conferences
                conferences = df_pandas['source_conference'].unique()
                num_conferences = len(conferences)
                
                if num_conferences == 0:
                    st.warning("No conference data available for visualization.")
                    return
                
                # Calculate subplot grid dimensions
                cols = min(2, num_conferences)  # Max 2 columns for better readability
                rows = (num_conferences + cols - 1) // cols  # Ceiling division
                
                # Create subplot titles for each conference
                subplot_titles = [f"{conf}" for conf in conferences]
                
                # Create subplots with Sankey diagrams
                fig = make_subplots(
                    rows=rows, 
                    cols=cols,
                    subplot_titles=subplot_titles,
                    specs=[[{"type": "sankey"} for _ in range(cols)] for _ in range(rows)],
                    vertical_spacing=0.1,
                    horizontal_spacing=0.05
                )
                
                # Color palette for continents
                continent_colors = {
                    'Asia': 'rgba(31, 119, 180, 0.8)',
                    'Europe': 'rgba(255, 127, 14, 0.8)', 
                    'North America': 'rgba(44, 160, 44, 0.8)',
                    'South America': 'rgba(214, 39, 40, 0.8)',
                    'Africa': 'rgba(148, 103, 189, 0.8)',
                    'Oceania': 'rgba(140, 86, 75, 0.8)'
                }
                
                # Plot each conference
                for i, conference in enumerate(conferences):
                    # Calculate subplot position
                    row = (i // cols) + 1
                    col = (i % cols) + 1
                    
                    # Filter data for this conference
                    conference_data = df_pandas[df_pandas['source_conference'] == conference]
                    
                    # Aggregate citation flows
                    flow_data = (conference_data
                            .groupby(['source_predominant_continent', 'cited_predominant_continent'])
                            ['paper_count'].sum().reset_index())
                    
                    if len(flow_data) == 0:
                        # Add empty sankey for conferences with no data
                        fig.add_trace(
                            go.Sankey(
                                node=dict(
                                    pad=15,
                                    thickness=20,
                                    line=dict(color="black", width=0.5),
                                    label=["No Data"],
                                    color=["rgba(128,128,128,0.5)"]
                                ),
                                link=dict(
                                    source=[],
                                    target=[],
                                    value=[]
                                )
                            ),
                            row=row, col=col
                        )
                        continue
                    
                    # Get unique continents for this conference
                    source_continents = flow_data['source_predominant_continent'].unique().tolist()
                    cited_continents = flow_data['cited_predominant_continent'].unique().tolist()
                    
                    # Create combined node list (source continents first, then cited continents)
                    # Add prefixes to distinguish source from cited
                    source_nodes = [f"Source: {cont}" for cont in source_continents]
                    cited_nodes = [f"Cited: {cont}" for cont in cited_continents]
                    all_nodes = source_nodes + cited_nodes
                    
                    # Create node index mapping
                    node_dict = {node: idx for idx, node in enumerate(all_nodes)}
                    
                    # Prepare Sankey data
                    source_indices = []
                    target_indices = []
                    values = []
                    
                    for _, row_data in flow_data.iterrows():
                        source_cont = row_data['source_predominant_continent']
                        cited_cont = row_data['cited_predominant_continent']
                        citation_count = row_data['paper_count']
                        
                        if citation_count > 0:  # Only include non-zero flows
                            source_idx = node_dict[f"Source: {source_cont}"]
                            target_idx = node_dict[f"Cited: {cited_cont}"]
                            
                            source_indices.append(source_idx)
                            target_indices.append(target_idx)
                            values.append(citation_count)
                    
                    # Create node colors based on continent
                    node_colors = []
                    for node in all_nodes:
                        # Extract continent name (remove "Source: " or "Cited: " prefix)
                        continent = node.split(": ", 1)[1] if ": " in node else node
                        node_colors.append(continent_colors.get(continent, 'rgba(128,128,128,0.8)'))
                    
                    # Create hover text
                    hover_text = []
                    for node in all_nodes:
                        node_type = "Source" if node.startswith("Source:") else "Cited"
                        continent = node.split(": ", 1)[1]
                        
                        if node_type == "Source":
                            total_outgoing = sum(values[j] for j, src in enumerate(source_indices) 
                                            if all_nodes[src] == node)
                            hover_text.append(f"{node_type}: {continent}<br>Total Citations Given: {total_outgoing:,}")
                        else:
                            total_incoming = sum(values[j] for j, tgt in enumerate(target_indices) 
                                            if all_nodes[tgt] == node)
                            hover_text.append(f"{node_type}: {continent}<br>Total Citations Received: {total_incoming:,}")
                    
                    # Add Sankey trace
                    fig.add_trace(
                        go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=15,
                                line=dict(color="black", width=0.5),
                                label=[node.split(": ", 1)[1] for node in all_nodes],  # Show only continent names
                                color=node_colors,
                                hovertemplate='%{customdata}<extra></extra>',
                                customdata=hover_text
                            ),
                            link=dict(
                                source=source_indices,
                                target=target_indices,
                                value=values,
                                hovertemplate='%{source.label} → %{target.label}<br>Citations: %{value:,}<extra></extra>'
                            )
                        ),
                        row=row, col=col
                    )
                
                # Update layout
                fig.update_layout(
                    title={
                        'text': "Citation Flow: Source Continents to Cited Continents by Conference",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16}
                    },
                    font_size=10,
                    height=500 * rows,  # Adjust height based on number of rows
                    margin=dict(l=50, r=50, t=80, b=50),
                    showlegend=False
                )
                
                # Display the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Required columns not found in dataframe. Expected: 'source_conference', 'source_predominant_continent', 'cited_predominant_continent', 'paper_count'")
                
        with tab2:
            if 'source_conference' in df_pandas.columns and 'source_predominant_continent' in df_pandas.columns:
                # Get unique conferences
                conferences = df_pandas['source_conference'].unique()
                num_conferences = len(conferences)
                
                if num_conferences == 0:
                    st.warning("No conference data available for visualization.")
                    return
                
                # Calculate subplot grid dimensions
                cols = min(2, num_conferences)  # Max 2 columns for better readability
                rows = (num_conferences + cols - 1) // cols  # Ceiling division
                
                # Create subplot titles for each conference
                subplot_titles = [f"{conf}" for conf in conferences]
                
                # Create subplots with Sankey diagrams
                fig = make_subplots(
                    rows=rows, 
                    cols=cols,
                    subplot_titles=subplot_titles,
                    specs=[[{"type": "sankey"} for _ in range(cols)] for _ in range(rows)],
                    vertical_spacing=0.1,
                    horizontal_spacing=0.05
                )
                
                # Color palette for continents
                continent_colors = {
                    'Asia': 'rgba(31, 119, 180, 0.8)',
                    'Europe': 'rgba(255, 127, 14, 0.8)', 
                    'North America': 'rgba(44, 160, 44, 0.8)',
                    'South America': 'rgba(214, 39, 40, 0.8)',
                    'Africa': 'rgba(148, 103, 189, 0.8)',
                    'Oceania': 'rgba(140, 86, 75, 0.8)'
                }

                df_pandas_percentage = (
                    df                  
                    .with_columns([
                        (
                            pl.col('paper_count') / 
                            pl.col('paper_count').sum().over(['source_conference']) * 100
                        )
                        .round(2)
                        .alias('paper_percentage')
                    ])
                    .sort(['source_predominant_continent'])                    
                ).to_pandas()
                
                # Plot each conference
                for i, conference in enumerate(conferences):
                    # Calculate subplot position
                    row = (i // cols) + 1
                    col = (i % cols) + 1
                    
                    # Filter data for this conference
                    conference_data = df_pandas_percentage[df_pandas_percentage['source_conference'] == conference]
                    
                    # Aggregate citation flows
                    flow_data = (conference_data
                            .groupby(['source_predominant_continent', 'cited_predominant_continent'])
                            ['paper_percentage'].sum().reset_index())
                    
                    if len(flow_data) == 0:
                        # Add empty sankey for conferences with no data
                        fig.add_trace(
                            go.Sankey(
                                node=dict(
                                    pad=15,
                                    thickness=20,
                                    line=dict(color="black", width=0.5),
                                    label=["No Data"],
                                    color=["rgba(128,128,128,0.5)"]
                                ),
                                link=dict(
                                    source=[],
                                    target=[],
                                    value=[]
                                )
                            ),
                            row=row, col=col
                        )
                        continue
                    
                    # Get unique continents for this conference
                    source_continents = flow_data['source_predominant_continent'].unique().tolist()
                    cited_continents = flow_data['cited_predominant_continent'].unique().tolist()
                    
                    # Create combined node list (source continents first, then cited continents)
                    # Add prefixes to distinguish source from cited
                    source_nodes = [f"Source: {cont}" for cont in source_continents]
                    cited_nodes = [f"Cited: {cont}" for cont in cited_continents]
                    all_nodes = source_nodes + cited_nodes
                    
                    # Create node index mapping
                    node_dict = {node: idx for idx, node in enumerate(all_nodes)}
                    
                    # Prepare Sankey data
                    source_indices = []
                    target_indices = []
                    values = []
                    
                    for _, row_data in flow_data.iterrows():
                        source_cont = row_data['source_predominant_continent']
                        cited_cont = row_data['cited_predominant_continent']
                        citation_count = row_data['paper_percentage']
                        
                        if citation_count > 0:  # Only include non-zero flows
                            source_idx = node_dict[f"Source: {source_cont}"]
                            target_idx = node_dict[f"Cited: {cited_cont}"]
                            
                            source_indices.append(source_idx)
                            target_indices.append(target_idx)
                            values.append(citation_count)
                    
                    # Create node colors based on continent
                    node_colors = []
                    for node in all_nodes:
                        # Extract continent name (remove "Source: " or "Cited: " prefix)
                        continent = node.split(": ", 1)[1] if ": " in node else node
                        node_colors.append(continent_colors.get(continent, 'rgba(128,128,128,0.8)'))
                    
                    # Create hover text
                    hover_text = []
                    for node in all_nodes:
                        node_type = "Source" if node.startswith("Source:") else "Cited"
                        continent = node.split(": ", 1)[1]
                        
                        if node_type == "Source":
                            total_outgoing = sum(values[j] for j, src in enumerate(source_indices) 
                                            if all_nodes[src] == node)
                            hover_text.append(f"{node_type}: {continent}<br>Total Citations Given: {total_outgoing:,}")
                        else:
                            total_incoming = sum(values[j] for j, tgt in enumerate(target_indices) 
                                            if all_nodes[tgt] == node)
                            hover_text.append(f"{node_type}: {continent}<br>Total Citations Received: {total_incoming:,}")
                    
                    # Add Sankey trace
                    fig.add_trace(
                        go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=15,
                                line=dict(color="black", width=0.5),
                                label=[node.split(": ", 1)[1] for node in all_nodes],  # Show only continent names
                                color=node_colors,
                                hovertemplate='%{customdata}<extra></extra>',
                                customdata=hover_text
                            ),
                            link=dict(
                                source=source_indices,
                                target=target_indices,
                                value=values,
                                hovertemplate='%{source.label} → %{target.label}<br>Citations: %{value:,}<extra></extra>'
                            )
                        ),
                        row=row, col=col
                    )
                
                # Update layout
                fig.update_layout(
                    title={
                        'text': "Citation Flow: Source Continents to Cited Continents by Conference",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16}
                    },
                    font_size=10,
                    height=500 * rows,  # Adjust height based on number of rows
                    margin=dict(l=50, r=50, t=80, b=50),
                    showlegend=False
                )
                
                # Display the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Required columns not found in dataframe. Expected: 'source_conference', 'source_predominant_continent', 'cited_predominant_continent', 'paper_count'")                
    
    def create_committee_visualizations(self, df: pl.DataFrame, do_ai_analysis: bool):
        """Create matplotlib visualizations for committee data"""
        if df.height == 0:
            return
                    
        if do_ai_analysis:
            self.print_llm_analysis(df, "committee")                        
                    
        df_pandas = df.to_pandas()
        
        # Create tabs for different visualizations
        if 'year' in df_pandas.columns:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Bar Chart (totals)", "📊 Bar Chart (percentages)", "📈 Line Chart(totals)", "📈 Line Chart(percentages)", "🥧 Pie Chart"])
        else:
            tab1, tab2, tab5 = st.tabs(["📊 Bar Chart (totals)", "📊 Bar Chart (percentages)", "🥧 Pie Chart"])
        
        with tab1:
            if 'conference' in df_pandas.columns:
                
                conference_data = df_pandas.groupby(['conference', 'continent'])['committee_count'].sum().reset_index()
                conferences = conference_data['conference'].unique()
                continents = conference_data['continent'].unique()

                # Create the plotly figure
                fig = go.Figure()

                # Add bars for each continent
                for continent in continents:
                    continent_data = conference_data[conference_data['continent'] == continent]
                    values = [continent_data[continent_data['conference'] == conf]['committee_count'].sum()
                            if conf in continent_data['conference'].values else 0
                            for conf in conferences]
                    
                    fig.add_trace(go.Bar(
                        name=continent,
                        x=conferences,
                        y=values,
                        opacity=0.8
                    ))

                # Update layout to match matplotlib styling
                fig.update_layout(
                    title='Committee members by conference and continent (totals)',
                    xaxis_title='Conference',
                    yaxis_title='Number of committee members',
                    barmode='group',  # This creates the grouped bar effect
                    width=1000,       # Equivalent to figsize=(14, 8)
                    height=571,       # Approximate height for 14x8 aspect ratio
                    xaxis=dict(
                        tickangle=45,  # Rotate x-axis labels
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)'
                    ),
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.01
                    )
                )

                # Display the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)  
                        
        with tab2:
            if 'conference' in df_pandas.columns:
                conference_data = (
                    df
                    .group_by("conference", "continent")
                    .agg([
                        pl.sum("committee_count")
                    ])                    
                    .with_columns([
                        (
                            pl.col('committee_count') / 
                            pl.col('committee_count').sum().over(['conference']) * 100
                        )
                        .round(2)
                        .alias('committee_percentage')
                    ])
                    .sort(['conference', 'continent'])                    
                ).to_pandas()
                conferences = conference_data['conference'].unique()
                continents = conference_data['continent'].unique()

                # Create the plotly figure
                fig = go.Figure()

                # Add bars for each continent
                for continent in continents:
                    continent_data = conference_data[conference_data['continent'] == continent]
                    values = [continent_data[continent_data['conference'] == conf]['committee_percentage'].sum()
                            if conf in continent_data['conference'].values else 0
                            for conf in conferences]
                    
                    fig.add_trace(go.Bar(
                        name=continent,
                        x=conferences,
                        y=values,
                        opacity=0.8
                    ))

                # Update layout to match matplotlib styling
                fig.update_layout(
                    title='Committee members by conference and continent (percentages)',
                    xaxis_title='Conference',
                    yaxis_title='Percentage of committee members',
                    barmode='group',  # This creates the grouped bar effect
                    width=1000,       # Equivalent to figsize=(14, 8)
                    height=571,       # Approximate height for 14x8 aspect ratio
                    xaxis=dict(
                        tickangle=45,  # Rotate x-axis labels
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)'
                    ),
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.01
                    )
                )

                # Display the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                        
        if 'year' in df_pandas.columns:
            with tab3:
                if 'conference' in df_pandas.columns and 'year' in df_pandas.columns:

                    conferences = df_pandas['conference'].unique()
                    num_conferences = len(conferences)

                    # Calculate subplot grid dimensions
                    cols = min(2, num_conferences)  # Max 2 columns (matching your original)
                    rows = (num_conferences + cols - 1) // cols  # Ceiling division

                    # Create subplot titles
                    subplot_titles = [f'Committee trends - {conf}' for conf in conferences]

                    # Create subplots
                    fig = make_subplots(
                        rows=rows, 
                        cols=cols,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.08,
                        horizontal_spacing=0.1
                    )

                    # Get all unique continents for consistent coloring
                    all_continents = df_pandas['continent'].unique()
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                    continent_colors = {continent: colors[i % len(colors)] for i, continent in enumerate(all_continents)}

                    # Plot each conference
                    for i, conference in enumerate(conferences):
                        row = i // cols + 1
                        col = i % cols + 1
                        
                        # Filter data for this conference
                        conference_data = df_pandas[df_pandas['conference'] == conference]
                        yearly_data = conference_data.groupby(['year', 'continent'])['committee_count'].sum().reset_index()
                        
                        # Plot each continent for this conference
                        for continent in yearly_data['continent'].unique():
                            continent_data = yearly_data[yearly_data['continent'] == continent]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=continent_data['year'],
                                    y=continent_data['committee_count'],
                                    mode='lines+markers',
                                    name=continent,
                                    line=dict(width=2, color=continent_colors[continent]),
                                    marker=dict(size=6, color=continent_colors[continent]),
                                    legendgroup=continent,  # Group legend entries by continent
                                    showlegend=(i == 0),    # Only show legend for first subplot
                                ),
                                row=row, col=col
                            )

                    # Update layout
                    fig.update_layout(
                        height=600 * rows,  # Equivalent to figsize height of 6 * rows
                        width=1000,         # Equivalent to figsize width of 14
                        title_text="Committee trends by conference (totals)",
                        title_x=0.5,        # Center the title
                        showlegend=True
                    )

                    # Update x and y axes for all subplots
                    fig.update_xaxes(
                        title_text="Year",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)',
                        tickangle=45
                    )

                    fig.update_yaxes(
                        title_text="Number of committee members",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)'
                    )
                    
                    # Display the plot in Streamlit
                    st.plotly_chart(fig, use_container_width=True)      
                    
            with tab4:
                if 'conference' in df_pandas.columns and 'year' in df_pandas.columns:                 
                    
                    df_pandas_percentages = (
                        df      
                        .with_columns([
                            (
                                pl.col('committee_count') / 
                                pl.col('committee_count').sum().over(['conference', 'year']) * 100
                            )
                            .round(2)
                            .alias('committee_percentage')
                        ])
                        .sort(['conference', 'continent', 'year'])                    
                    ).to_pandas()                                    
                    conferences = df_pandas['conference'].unique()
                    num_conferences = len(conferences)

                    # Calculate subplot grid dimensions
                    cols = min(2, num_conferences)  # Max 2 columns (matching your original)
                    rows = (num_conferences + cols - 1) // cols  # Ceiling division

                    # Create subplot titles
                    subplot_titles = [f'Committee trends - {conf}' for conf in conferences]

                    # Create subplots
                    fig = make_subplots(
                        rows=rows, 
                        cols=cols,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.08,
                        horizontal_spacing=0.1
                    )

                    # Get all unique continents for consistent coloring
                    all_continents = df_pandas_percentages['continent'].unique()
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                    continent_colors = {continent: colors[i % len(colors)] for i, continent in enumerate(all_continents)}

                    # Plot each conference
                    for i, conference in enumerate(conferences):
                        row = i // cols + 1
                        col = i % cols + 1
                        
                        # Filter data for this conference
                        conference_data = df_pandas_percentages[df_pandas_percentages['conference'] == conference]
                        yearly_data = conference_data.groupby(['year', 'continent'])['committee_percentage'].sum().reset_index()
                        
                        # Plot each continent for this conference
                        for continent in yearly_data['continent'].unique():
                            continent_data = yearly_data[yearly_data['continent'] == continent]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=continent_data['year'],
                                    y=continent_data['committee_percentage'],
                                    mode='lines+markers',
                                    name=continent,
                                    line=dict(width=2, color=continent_colors[continent]),
                                    marker=dict(size=6, color=continent_colors[continent]),
                                    legendgroup=continent,  # Group legend entries by continent
                                    showlegend=(i == 0),    # Only show legend for first subplot
                                ),
                                row=row, col=col
                            )

                    # Update layout
                    fig.update_layout(
                        height=600 * rows,  # Equivalent to figsize height of 6 * rows
                        width=1000,         # Equivalent to figsize width of 14
                        title_text="Committee trends by conference (percentages)",
                        title_x=0.5,        # Center the title
                        showlegend=True
                    )

                    # Update x and y axes for all subplots
                    fig.update_xaxes(
                        title_text="Year",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)',
                        tickangle=45
                    )

                    fig.update_yaxes(
                        title_text="Percentage of committee members",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.3)'
                    )
                    
                    # Display the plot in Streamlit
                    st.plotly_chart(fig, use_container_width=True)                         
                    
                            
                    
        with tab5:
            if 'conference' in df_pandas.columns:
                
                # self.create_matplotlib_chart(fig, "Committee distribution by conference and continent")    
                conferences = df_pandas['conference'].unique()

                # Calculate number of rows and columns for subplots
                n_conferences = len(conferences)
                n_cols = min(3, n_conferences)  # Max 3 columns
                n_rows = (n_conferences + n_cols - 1) // n_cols  # Ceiling division

                # Create subplot titles
                subplot_titles = []
                for conference in conferences:
                    conf_data = df_pandas[df_pandas['conference'] == conference]
                    total_committees = conf_data['committee_count'].sum()
                    subplot_titles.append(f'{conference}<br>({total_committees} committees)')

                # Create subplots with pie chart specs
                VERTICAL_SPACING = 0
                HORIZONTAL_SPACING = 0.1
                fig = make_subplots(
                    rows=n_rows, 
                    cols=n_cols,
                    subplot_titles=subplot_titles,
                    specs=[[{"type": "pie"}] * n_cols for _ in range(n_rows)],
                    vertical_spacing=VERTICAL_SPACING,
                    horizontal_spacing=HORIZONTAL_SPACING
                )

                # Define colors (equivalent to plt.cm.Set3)
                colors = [
                    '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
                    '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'
                ]

                # Plot each conference
                for i, conference in enumerate(conferences):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    
                    # Filter data for this conference
                    conf_data = df_pandas[df_pandas['conference'] == conference]
                    continent_totals = conf_data.groupby('continent')['committee_count'].sum()
                    
                    if len(continent_totals) > 0:
                        # Calculate percentages
                        total = continent_totals.sum()
                        percentages = [(value/total)*100 for value in continent_totals.values]
                        
                        # Create custom hover text with percentages
                        hover_text = [f'{continent}<br>{value} committees<br>{percent:.1f}%' 
                                    for continent, value, percent in 
                                    zip(continent_totals.index, continent_totals.values, percentages)]
                        
                        # Create custom legend labels with percentages
                        legend_labels = [f'{continent}: {percent:.1f}%' 
                                        for continent, percent in zip(continent_totals.index, percentages)]
                        
                        # Select colors for this conference's continents
                        conf_colors = colors[:len(continent_totals)]
                        
                        fig.add_trace(
                            go.Pie(
                                labels=legend_labels,  # Use percentage labels
                                values=continent_totals.values,
                                name=conference,
                                marker=dict(colors=conf_colors),
                                hovertemplate='%{label}<br>%{value} committees<br>%{percent}<extra></extra>',
                                textinfo='none',  # Don't show text on pie slices
                                showlegend=True,
                                legendgroup=f'group{i}',  # Separate legend for each pie
                                legend=f'legend{i+1}' if i > 0 else 'legend',  # Assign to specific legend
                                domain=dict(row=row-1, column=col-1)  # Position the pie
                            ),
                            row=row, col=col
                        )
                    else:
                        # For conferences with no data, add empty pie with annotation
                        fig.add_trace(
                            go.Pie(
                                labels=['No data'],
                                values=[1],
                                marker=dict(colors=['lightgray']),
                                showlegend=False,
                                textinfo='none',
                                hoverinfo='skip'
                            ),
                            row=row, col=col
                        )
                        
                        # Add annotation for "No data"
                        fig.add_annotation(
                            text="No data",
                            x=0.5, y=0.5,
                            xref=f"x{i+1 if i > 0 else ''}", 
                            yref=f"y{i+1 if i > 0 else ''}",
                            showarrow=False,
                            font=dict(size=12, color="black"),
                            align="center"
                        )

                # Create legend configurations for each subplot
                legend_configs = {}
                for i, conference in enumerate(conferences):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    legend_x, legend_y = self.get_legend_position(row, col, n_rows, n_cols)
                    
                    legend_key = f'legend{i+1}' if i > 0 else 'legend'
                    legend_configs[legend_key] = dict(
                        x=legend_x,
                        y=legend_y,
                        xanchor="left",
                        # yanchor="middle",
                        yanchor="top",
                        font=dict(size=9),
                        bgcolor="rgba(255,255,255,0.5)",
                        bordercolor="rgba(0,0,0,0.1)",
                        borderwidth=1
                    )

                # Update layout
                fig.update_layout(
                    title=dict(
                        text='Distribution of committee members by continent',
                        x=0.5,
                        font=dict(size=16, color="black")
                    ),
                    height=600 * n_rows,  # Equivalent to 6 * n_rows
                    width=900 * n_cols,   # Increased width to accommodate individual legends
                    # margin=dict(l=50, r=100, t=80, b=50),  # Adjust margins
                    **legend_configs  # Add all legend configurations
                )           
                
                # Display the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)
    
    def create_committee_country_visualizations(self, df: pl.DataFrame, do_ai_analysis: bool):
        """Create matplotlib visualizations for committee country data"""
        if df.height == 0:
            return
        
        if do_ai_analysis:
            self.print_llm_analysis(df, "committee")                
            
        df_pandas = df.to_pandas()
        
        print("aqui 2")
        print(df_pandas.columns)
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["📊 Bar Chart", "📈 Line Chart"])
        
        with tab1:
            if 'conference' in df_pandas.columns and 'committee_country' in df_pandas.columns:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Group by conference and country
                conference_data = df_pandas.groupby(['conference', 'committee_country'])['committee_count'].sum().reset_index()
                
                # Get top countries to avoid overcrowding
                top_countries = df_pandas.groupby('committee_country')['committee_count'].sum().nlargest(10).index
                filtered_data = conference_data[conference_data['committee_country'].isin(top_countries)]
                
                conferences = filtered_data['conference'].unique()
                countries = filtered_data['committee_country'].unique()
                
                x = np.arange(len(conferences))
                width = 0.8 / len(countries)
                
                for i, country in enumerate(countries):
                    country_data = filtered_data[filtered_data['committee_country'] == country]
                    values = [country_data[country_data['conference'] == conf]['committee_count'].sum() 
                             if conf in country_data['conference'].values else 0 
                             for conf in conferences]
                    
                    ax.bar(x + i * width, values, width, label=country, alpha=0.8)
                
                ax.set_xlabel('Conference')
                ax.set_ylabel('Number of Committee Members')
                ax.set_title('Committee Members by Conference and Country (Top 10 Countries)')
                ax.set_xticks(x + width * (len(countries) - 1) / 2)
                ax.set_xticklabels(conferences, rotation=45)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                self.create_matplotlib_chart(fig, "Committee Members by Conference and Country")
        
        with tab2:
            if 'year' in df_pandas.columns and 'committee_country' in df_pandas.columns:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Group and sort data properly for line chart
                yearly_data = df_pandas.groupby(['year', 'committee_country'])['committee_count'].sum().reset_index()
                
                # Sort by year to ensure proper line progression
                yearly_data = yearly_data.sort_values('year')
                
                # Get top countries and years
                top_countries = df_pandas.groupby('committee_country')['committee_count'].sum().nlargest(8).index
                filtered_yearly = yearly_data[yearly_data['committee_country'].isin(top_countries)]
                
                countries = sorted(filtered_yearly['committee_country'].unique())
                years = sorted(filtered_yearly['year'].unique())
                
                # Plot line for each country
                for country in countries:
                    country_data = filtered_yearly[filtered_yearly['committee_country'] == country]
                    
                    # Create complete year series (fill missing years with 0)
                    country_series = pd.DataFrame({'year': years})
                    country_series = country_series.merge(
                        country_data[['year', 'committee_count']], 
                        on='year', 
                        how='left'
                    ).fillna(0)
                    
                    # Sort again to be absolutely sure
                    country_series = country_series.sort_values('year')
                    
                    ax.plot(country_series['year'], country_series['committee_count'], 
                           marker='o', linewidth=2, markersize=6, label=country)
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of Committee Members')
                ax.set_title('Committee Trends Over Time by Country (Top 8 Countries)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
                # Set x-axis to show all years as integers
                ax.set_xticks(years)
                ax.set_xticklabels([str(int(year)) for year in years], rotation=45)
                
                plt.tight_layout()
                
                self.create_matplotlib_chart(fig, "Committee Trends Over Time by Country")

    def _call_ollama(self, model_name: str, system_prompt: str, user_content: str) -> str | None:
        """Make a call to Ollama with error handling"""
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                options={"temperature": 0},
                keep_alive='30m'
            )
            return response.message.content
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
        
    def _format_dataframe_to_markdown(self, df: pl.DataFrame):
        """Convert DataFrame to markdown with proper formatting"""
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_rows=-1,
            tbl_cols=-1,
            fmt_str_lengths=999999,
            tbl_width_chars=100,
        ):
            return str(df.head(9999))
        
    def _phase1_individual_analysis(self, df: pl.DataFrame, prompt_case: str) -> dict:
        """
        Phase 1: Analyze each conference individually
        
        Args:
            df: Polars DataFrame with conference data
            
        Returns:
            Dictionary with conference names as keys and analysis results as values
        """
        SYSTEM_PROMPT = f"""
        You are a data analysis assistant specialized in research publication trends.
        You will receive structured tabular or Markdown data {self.PROMPTS[prompt_case]}

        ## GOAL ##
        1. Carefully interpret the data—analyze totals, proportions, changes over time, and differences between categories.
        2. Identify clear trends (growth, decline, stability), notable peaks or drops, and distribution patterns.
        3. Communicate findings in two parts:
        - **Concise bullet points** for quick insights.
        - **Brief narrative summary** to explain the patterns.

        ## RULES ##
        - If data spans multiple years, highlight temporal trends.
        - Do not fabricate data—base all insights solely on the provided dataset.
        - Keep tone analytical but accessible to non-specialists.
        - Always output **all sections in the exact order of the template** below.
        - Even if a section has no findings, explicitly state "No significant findings" for that section.

        ## OUTPUT TEMPLATE (MANDATORY) ##
        1. Trends
        2. Peaks and drops
        3. Continental distribution
        4. Unknown and other continents

        ## IMPORTANT ##
        - You must strictly adhere to the output structure above.
        - Don't mention numbers.
        - No extra sections, no omissions, no reordering.
        
        ## CRITICAL NUMBER HANDLING ##
        - Always write complete 4-digit years (e.g., 2023, not 202)
        - Double-check all numbers before finalizing your response
        - If you see a year like "202", it should be "2023" or similar        
        """

        results = {}
        
        print("=== PHASE 1: Individual Conference Analysis ===\n")
        
        conference_column = "conference" if prompt_case == "committee" else "source_conference" 
        for conf, group_df in df.group_by(conference_column):
            if prompt_case == 'paper':
                pivot_df = group_df.pivot(values="paper_count", index="source_year", on="source_predominant_continent", aggregate_function="sum").with_columns(
                    pl.sum_horizontal(pl.exclude("source_year")).alias("Total")
                )
            elif prompt_case == 'citation':
                pivot_df = group_df.pivot(values="paper_count", index="source_year", on="cited_predominant_continent", aggregate_function="sum").with_columns(
                    pl.sum_horizontal(pl.exclude("source_year")).alias("Total")
                )                
            elif prompt_case == 'paper_and_citation':
                pivot_df = group_df
            else:
                pivot_df = group_df.pivot(values="committee_count", index="year", on="continent", aggregate_function="sum").with_columns(
                    pl.sum_horizontal(pl.exclude("year")).alias("Total")
                )                     

            print(f"--- Analyzing {conf} ---")

            data_str = self._format_dataframe_to_markdown(pivot_df)
            print(data_str)
            user_content = f"Conference: {conf}\nMarkdown table:\n{data_str}"
            response = self._call_ollama("gemma3:4b-it-q4_K_M",SYSTEM_PROMPT, user_content)
            results[conf] = str(response)
        
        return results
    
    def _phase2_comparative_analysis(self, individual_results: Optional[dict[str, str]] = None) -> str:
        """
        Phase 2: Compare all conferences and provide aggregated insights
        
        Args:
            individual_results: Optional dict of individual results. If None, uses stored results.
            
        Returns:
            Comparative analysis string
        """
        SYSTEM_PROMPT = """
        You are a data analysis assistant specialized in comparative research publication analysis.
        You will receive analysis results from multiple conferences that have already been individually analyzed.

        ## GOAL ##
        1. Compare and contrast findings across all conferences.
        2. Identify cross-conference patterns, similarities, and differences.
        3. Provide aggregated insights about the overall research landscape.
        4. Highlight which conferences show similar or different behaviors.

        ## RULES ##
        - Focus on comparative insights rather than repeating individual conference details.
        - Identify overarching patterns that emerge when looking at all conferences together.
        - Note which conferences are outliers or follow different patterns.
        - Keep tone analytical but accessible to non-specialists.
        - Always output **all sections in the exact order of the template** below.
        - Even if a section has no findings, explicitly state "No significant findings" for that section.

        ## OUTPUT TEMPLATE (MANDATORY) ##
        1. Trends
        2. Peaks and drops
        3. Continental distribution
        4. Unknown and other continents

        ## IMPORTANT ##
        - You must strictly adhere to the output structure above.
        - No extra sections, no omissions, no reordering.
        - Focus on COMPARATIVE and AGGREGATED insights across all conferences.
        
        ## CRITICAL NUMBER HANDLING ##
        - Always write complete 4-digit years (e.g., 2023 and 2015, not 202 and 201)
        - Double-check all numbers before finalizing your response
        - If you see a year like "202", it should be "2023" or similar        
        """
        
        # if individual_results is None:
        #     individual_results = self.conference_results
        
        if not individual_results:
            return "No individual results available for comparison."
        
        print("=== PHASE 2: Pairwise Comparative Analysis ===\n")
        
        # Convert to list for easier pairwise processing
        results_list = list(individual_results.items())
        
        if len(results_list) == 1:
            print("Only one conference found. Returning individual result.")
            return list(individual_results.values())[0]
        
        round_number = 1
        
        # Continue until we have only one result left
        while len(results_list) > 1:
            print(f"--- Round {round_number} of Pairwise Comparisons ---")
            new_results = []
            
            # Process pairs
            for i in range(0, len(results_list), 2):
                if i + 1 < len(results_list):
                    # We have a pair
                    conf1, analysis1 = results_list[i]
                    conf2, analysis2 = results_list[i + 1]
                    
                    print(f"Comparing: {conf1} vs {conf2}")
                    
                    # Create pairwise comparison content
                    pair_content = f"""Compare these two conference analyses and provide aggregated insights:

                        ## Conference {conf1}:
                        {analysis1}

                        ---

                        ## Conference {conf2}:
                        {analysis2}

                        Please provide a comparative analysis that consolidates insights from both conferences."""
                    
                    # Get pairwise comparison
                    pair_result = self._call_ollama("gemma3:4b-it-q4_K_M", SYSTEM_PROMPT, pair_content)
                    
                    # Create a name for this comparison result
                    combined_name = f"{conf1}_vs_{conf2}"
                    new_results.append((combined_name, pair_result))
                
                else:
                    # Odd one out - carry forward to next round
                    new_results.append(results_list[i])
                    print(f"Carrying forward: {results_list[i][0]} (no pair in this round)")
            
            results_list = new_results
            round_number += 1
            print(f"Round {round_number - 1} complete. {len(results_list)} results remaining.\n")
        
        # Final result
        final_name, final_result = results_list[0]
        
        print("=== FINAL COMPARATIVE ANALYSIS ===")
        print(f"Final analysis combines all conferences through pairwise comparisons")
        return final_result
        
    def print_llm_analysis(self, df: pl.DataFrame, prompt_case: str):
        SYSTEM_PROMPT = """
        You are a data analysis assistant specialized in research publication trends.
        You will receive structured tabular or Markdown data containing counts of papers published,
        categorized by continent, conference, and year.

        ## GOAL ##
        1. Carefully interpret the data—analyze totals, proportions, changes over time, and differences between categories.
        2. Identify clear trends (growth, decline, stability), notable peaks or drops, and distribution patterns.
        3. Communicate findings in two parts:
        - **Concise bullet points** for quick insights.
        - **Brief narrative summary** to explain the patterns.

        ## RULES ##
        - If data spans multiple years, highlight temporal trends.
        - Do not fabricate data—base all insights solely on the provided dataset.
        - Keep tone analytical but accessible to non-specialists.
        - Always output **all sections in the exact order of the template** below.
        - Even if a section has no findings, explicitly state "No significant findings" for that section.

        ## OUTPUT TEMPLATE (MANDATORY) ##
        1. Trends
        2. Peaks and drops
        3. Continental distribution
        4. Unknown and other continents

        ## IMPORTANT ##
        - You must strictly adhere to the output structure above.
        - No extra sections, no omissions, no reordering.
        
        ## CRITICAL NUMBER HANDLING ##
        - Always write complete 4-digit years (e.g., 2022 and 2015, not 202 and 201)
        - Double-check all numbers before finalizing your response
        - If you see a year like "202" or "201", it should be "2021" or "2015" or similar             
        """
        # SYSTEM_PROMPT = """
        # You are a data analysis assistant specialized in research publication trends.
        # You will receive structured tabular or HTML data containing counts of papers published,
        # categorized by continent, conference, and year.

        # Your role is to:
        # 1. Interpret the data carefully—look at totals, proportions, changes over time, and differences between categories.
        # 2. Identify trends (growth, decline, stability), notable peaks or drops, and patterns in the distribution.
        # 3. Compare categories (e.g., which continent leads in certain years, which conferences have steady or explosive growth, which copnference has more papers of each continent).
        # 4. Communicate clearly—provide concise bullet points for quick insight, followed by a short narrative summary.
        # Rules:
        # - If data covers multiple years, highlight temporal trends.
        # - If a category is missing or has incomplete data, note it explicitly.
        # - Do not invent data—only use what is provided.
        # - Keep the tone analytical yet accessible.
        # """
        
        individual_results = self._phase1_individual_analysis(df=df, prompt_case=prompt_case)
        final_result = self._phase2_comparative_analysis(individual_results)
        if hasattr(st, 'markdown'):
            st.markdown(final_result)
        else:
            print(final_result)
        
        # data_str = ""
        # for conf, group_df in df.group_by("source_conference"):
        #     print(f"--- {conf} ---")
        #     data_str = self._format_dataframe_to_markdown(group_df)
        #     print(data_str)

        #     response = self._call_ollama(model_name="gemma3:4b", system_prompt=SYSTEM_PROMPT, user_content=f"Markdown table:\n{data_str}")
        #     st.markdown(response)

        
        # data_str = json.dumps(df.to_dicts(), indent=2)
        # response = ollama.chat(
        #     model="mistral",
        #     messages=[
        #         {"role": "system", "content": SYSTEM_PROMPT},
        #         {"role": "user", "content": f"Here is the dataset:\n{data_str}\n\nAnalyze it and provide insights. /no_think"}
        #     ]
        # )        
        # st.markdown(response.message.content)

    # Calculate legend positions for each subplot
    def get_legend_position(self, row, col, n_rows, n_cols):
        # Calculate the center position of each subplot
        subplot_width = 1.0 / n_cols
        subplot_height = 1.0 / n_rows
        
        # Legend position relative to subplot
        x_center = (col - 0.5) * subplot_width 
        y_center = 1 - (row - 0.5) * subplot_height
        
        # Position legend to the right of the subplot
        legend_x = x_center + subplot_width * 0.3
        # legend_y = y_center
        legend_y = y_center - subplot_height * 0.3
        
        return legend_x, legend_y    
    
    def run_analytics_queries(self):
        """Run analytics queries based on selected analysis type"""
        if not st.session_state.analytics_client:
            st.error("Please initialize the database connection first!")
            return
        
        # st.header("📊 Analytics Dashboard")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "Papers by Conference, Continent and Year",
                # "Papers by Conference and Continent",
                "Citations by Conference, Continent and Year",
                # "Citations by Conference and Continent",                
                "Citations by Conference and Source and Cited Continent",                
                # "Committees by Conference, Country and Year", 
                "Committees by Continent and Year",
                "Committees by Continent"
            ],
            key="analysis_type_selector"
        )
        
        # Show description of selected analysis
        analysis_descriptions = {
            "Papers by Conference, Continent and Year": "Analyze paper counts across conferences, continents and years",
            # "Papers by Conference and Continent": "Compare paper distribution by conference and continent",
            "Citations by Conference, Continent and Year": "Analyze citation counts across conferences, continents and years",
            # "Citations by Conference and Continent": "Compare citation distribution by conference and continent",            
            "Citations by Conference and Source and Cited Continent": "Compare citation distribution by conference and source and cited continent",            
            # "Committees by Conference, Country and Year": "Track committee member distribution by conference, country, and year",
            "Committees by Continent and Year": "Analyze committee member trends across continents over time",
            "Committees by Continent": "Overview of committee member distribution by continent"            
        }
        
        if analysis_type in analysis_descriptions:
            st.info(analysis_descriptions[analysis_type])
        
        # Get filters based on selected analysis type
        filters, do_ai_analysis = self.render_filters_sidebar(analysis_type)
        
        if st.button("🚀 Run Analysis", type="primary"):
            try:
                with st.spinner("Running analysis..."):
                    if analysis_type == "Papers by Conference, Continent and Year":
                        result = st.session_state.analytics_client.query_paper_count_per_conference_continent_and_year(
                            text=filters['text'],
                            conferences=filters['conferences'],
                            years=filters['years'],
                            continents=filters['continents']
                        )
                        self.display_dataframe_with_download(result, "Papers by Conference, Continent and Year", "papers_conf_cont_year")
                        self.create_paper_visualizations(result, do_ai_analysis)
                        
                    elif analysis_type == "Papers by Conference and Continent":
                        result = st.session_state.analytics_client.query_paper_count_per_conference_and_continent(
                            text=filters['text'],                            
                            conferences=filters['conferences'],
                            continents=filters['continents']
                        )
                        self.display_dataframe_with_download(result, "Papers by Conference and Continent", "papers_conf_cont")
                        self.create_paper_visualizations(result, do_ai_analysis)
                        
                    if analysis_type == "Citations by Conference, Continent and Year":
                        result = st.session_state.analytics_client.query_citation_count_per_conference_continent_and_year(
                            text=filters['text'],
                            conferences=filters['conferences'],
                            years=filters['years'],
                            continents=filters['continents'],
                            cited_continents=filters['cited_continents'],
                        )
                        self.display_dataframe_with_download(result, "Papers by Conference, Continent and Year", "papers_conf_cont_year")
                        self.create_citation_visualizations(result, do_ai_analysis)
                        
                    elif analysis_type == "Citations by Conference and Continent":
                        result = st.session_state.analytics_client.query_citation_count_per_conference_and_continent(
                            text=filters['text'],                            
                            conferences=filters['conferences'],
                            continents=filters['continents'],
                            cited_continents=filters['cited_continents'],
                        )
                        self.display_dataframe_with_download(result, "Papers by Conference and Continent", "papers_conf_cont")
                        self.create_citation_visualizations(result, do_ai_analysis)                        
                        
                    elif analysis_type == "Citations by Conference and Source and Cited Continent":
                        result = st.session_state.analytics_client.query_citation_count_per_conference_source_continent_and_year(
                            text=filters['text'],                            
                            conferences=filters['conferences'],
                            continents=filters['continents'],
                            cited_continents=filters['cited_continents'],
                        )
                        self.display_dataframe_with_download(result, "Papers by Conference and Continent", "papers_conf_cont")
                        self.create_citation_by_source_and_cited_continent_visualizations(result, do_ai_analysis)                        
                                                
                        
                    elif analysis_type == "Committees by Conference, Country and Year":
                        result = st.session_state.analytics_client.get_committees_per_conference_country_year_count(
                            conferences=filters['conferences'],
                            years=filters['years']
                        )
                        self.display_dataframe_with_download(result, "Committees by Conference, Country and Year", "committees_conf_country_year")
                        self.create_committee_country_visualizations(result, do_ai_analysis)
                        
                    elif analysis_type == "Committees by Continent and Year":
                        result = st.session_state.analytics_client.get_committees_per_continent_year_count(
                            conferences=filters['conferences'],
                            continents=filters['continents'],
                            years=filters['years']
                        )
                        self.display_dataframe_with_download(result, "Committees by Continent and Year", "committees_cont_year")
                        self.create_committee_visualizations(result, do_ai_analysis)
                        
                    elif analysis_type == "Committees by Continent":
                        result = st.session_state.analytics_client.get_committees_per_continent_count(
                            conferences=filters['conferences'],
                            continents=filters['continents']
                        )
                        self.display_dataframe_with_download(result, "Committees by Continent", "committees_cont")
                        self.create_committee_visualizations(result, do_ai_analysis)
                        
            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")
                st.exception(e)
    
    def run(self):
        """Main application runner"""
        st.title("Paper Analytics Dashboard")
        st.markdown("---")
        
        # Setup connection status display
        self.setup_connection()
        
        # Run analytics (filters are rendered inside based on analysis type)
        self.run_analytics_queries()

# Run the application
if __name__ == "__main__":
    app = StreamlitPaperAnalytics()
    app.run()