import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl
from typing import Optional, List
import numpy as np
import io

# Your specific imports and initialization
import src.papers.domain.multimodal_paper_query as mpq
from src.papers.domain.multimodal_paper_query import Conference, FilterCondition
from src.papers.io.db import Milvus, Neo4j, SQLite
import os
import pycountry_convert as pc

# Import your analytics class (adjust path as needed)
from src.papers.domain.paper_analytics import PaperAnalytics, Continent
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

class StreamlitPaperAnalytics:
    """Streamlit interface for Paper Analytics using matplotlib"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.setup_matplotlib_style()
        
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
            
    def initialize_clients(self):
        """Initialize database clients using environment variables"""
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
            
            # Initialize clients
            milvus_client = Milvus(
                collection=MILVUS_COLLECTION,
                alias=MILVUS_ALIAS,
                host=MILVUS_HOST,
                port=int(MILVUS_PORT) if MILVUS_PORT else 19530,
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
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize clients: {str(e)}")
            return False
    
    def setup_connection(self):
        """Setup database connection section"""
        st.sidebar.header("🔗 Database Connection")
        
        # Show environment variable status
        env_vars = [
            "MILVUS_COLLECTION", "MILVUS_ALIAS", "MILVUS_HOST", "MILVUS_PORT",
            "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"
        ]
        
        missing_vars = [var for var in env_vars if not os.getenv(var)]
        
        if missing_vars:
            st.sidebar.warning(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
            
            # Manual configuration option
            with st.sidebar.expander("Manual Configuration"):
                for var in missing_vars:
                    if var.endswith("PASSWORD"):
                        os.environ[var] = st.text_input(f"{var}:", type="password", key=f"manual_{var}")
                    elif var.endswith("PORT"):
                        os.environ[var] = str(st.number_input(f"{var}:", value=19530 if "MILVUS" in var else 7687, key=f"manual_{var}"))
                    else:
                        os.environ[var] = st.text_input(f"{var}:", key=f"manual_{var}")
        else:
            st.sidebar.success("✅ All environment variables found")
        
        if st.sidebar.button("🔌 Initialize Connection"):
            if self.initialize_clients():
                st.sidebar.success("✅ Connected successfully!")
            else:
                st.sidebar.error("❌ Connection failed!")
    
    def render_filters_sidebar(self, analysis_type=None):
        """Render filter controls in sidebar based on selected analysis type"""
        st.sidebar.header("🔍 Filters")
        
        # Define which filters are needed for each analysis type
        filter_requirements = {
            "Papers by Conference, Continent & Year": ["conferences", "years", "continents"],
            "Papers by Conference & Continent": ["conferences", "continents"],
            "Committees by Conference, Country & Year": ["conferences", "years"],
            "Committees by Continent & Year": ["conferences", "continents", "years"],
            "Committees by Continent": ["conferences", "continents"]
        }
        
        filters = {}
        
        if analysis_type and analysis_type in filter_requirements:
            required_filters = filter_requirements[analysis_type]
            st.sidebar.info(f"Filters for: **{analysis_type}**")
        else:
            required_filters = ["conferences", "years", "continents"]  # Default: show all
            if not analysis_type:
                st.sidebar.info("Select an analysis type to see relevant filters")
        
        # Conference filter - using your Conference enum
        if "conferences" in required_filters:
            available_conferences = [conf.value for conf in Conference]
            conferences = st.sidebar.multiselect(
                "Select Conferences",
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
            current_year = 2024
            years = st.sidebar.multiselect(
                "Select Years",
                options=[str(year) for year in range(2010, current_year + 1)],
                key="selected_years",
                help="Filter by specific years"
            )
            filters['years'] = years if years else None
        else:
            filters['years'] = None
        
        # Continent filter
        if "continents" in required_filters:
            continents = st.sidebar.multiselect(
                "Select Continents",
                options=["Asia", "Europe", "North America", "South America", "Africa", "Oceania", "Antarctica"],
                key="selected_continents",
                help="Filter by specific continents"
            )
            filters['continents'] = continents if continents else None
        else:
            filters['continents'] = None
        
        # Show a summary of active filters
        active_filters = [k for k, v in filters.items() if v is not None and len(v) > 0]
        if active_filters:
            st.sidebar.success(f"✅ Active filters: {', '.join(active_filters)}")
        else:
            st.sidebar.info("ℹ️ No filters applied (showing all data)")
        
        return filters
    
    def display_dataframe_with_download(self, df: pl.DataFrame, title: str, key: str):
        """Display DataFrame with download option"""
        st.subheader(f"📋 {title}")
        
        if df.height > 0:
            # Convert to pandas for better display
            df_pandas = df.to_pandas()
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.height)
            with col2:
                st.metric("Total Columns", df.width)
            with col3:
                if 'paper_count' in df.columns:
                    st.metric("Total Papers", df.select(pl.sum('paper_count')).item())
                elif 'committee_count' in df.columns:
                    st.metric("Total Committees", df.select(pl.sum('committee_count')).item())
            
            # Display table
            st.dataframe(df_pandas, use_container_width=True, key=f"df_{key}")
            
            # Download button
            csv = df_pandas.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {title} as CSV",
                data=csv,
                file_name=f"{title.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"download_{key}"
            )
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
    
    def create_paper_visualizations(self, df: pl.DataFrame):
        """Create matplotlib visualizations for paper data"""
        if df.height == 0:
            return
            
        df_pandas = df.to_pandas()
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Bar Chart", "📈 Line Chart", "🥧 Pie Chart", "🔥 Heatmap"])
        
        with tab1:
            if 'source_year' in df_pandas.columns and 'source_conference' in df_pandas.columns:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                grouped_data = df_pandas.groupby(['source_conference', 'source_predominant_continent'])['paper_count'].sum().reset_index()
                
                # Create grouped bar chart
                conferences = grouped_data['source_conference'].unique()
                continents = grouped_data['source_predominant_continent'].unique()
                
                x = np.arange(len(conferences))
                width = 0.8 / len(continents)
                
                for i, continent in enumerate(continents):
                    continent_data = grouped_data[grouped_data['source_predominant_continent'] == continent]
                    values = [continent_data[continent_data['source_conference'] == conf]['paper_count'].sum() 
                             if conf in continent_data['source_conference'].values else 0 
                             for conf in conferences]
                    
                    ax.bar(x + i * width, values, width, label=continent, alpha=0.8)
                
                ax.set_xlabel('Conference')
                ax.set_ylabel('Number of Papers')
                ax.set_title('Papers by Conference and Continent')
                ax.set_xticks(x + width * (len(continents) - 1) / 2)
                ax.set_xticklabels(conferences, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                self.create_matplotlib_chart(fig, "Papers by Conference and Continent")
        
        with tab2:
            if 'source_year' in df_pandas.columns:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                yearly_data = df_pandas.groupby(['source_year', 'source_predominant_continent'])['paper_count'].sum().reset_index()
                
                for continent in yearly_data['source_predominant_continent'].unique():
                    continent_data = yearly_data[yearly_data['source_predominant_continent'] == continent]
                    ax.plot(continent_data['source_year'], continent_data['paper_count'], 
                           marker='o', linewidth=2, markersize=6, label=continent)
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of Papers')
                ax.set_title('Paper Trends Over Time by Continent')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                self.create_matplotlib_chart(fig, "Paper Trends Over Time")
        
        with tab3:
            if 'source_conference' in df_pandas.columns:
                conferences = df_pandas['source_conference'].unique()
                
                # Calculate number of rows and columns for subplots
                n_conferences = len(conferences)
                n_cols = min(3, n_conferences)  # Max 3 columns
                n_rows = (n_conferences + n_cols - 1) // n_cols  # Ceiling division
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
                
                # Handle case where there's only one subplot
                if n_conferences == 1:
                    axes = [axes]
                elif n_rows == 1 and n_cols > 1:
                    axes = axes.flatten()
                elif n_rows > 1:
                    axes = axes.flatten()
                
                colors = plt.cm.Set3(np.linspace(0, 1, 10))  # Generate enough colors
                
                for i, conference in enumerate(conferences):
                    # Filter data for this conference
                    conf_data = df_pandas[df_pandas['source_conference'] == conference]
                    continent_totals = conf_data.groupby('source_predominant_continent')['paper_count'].sum()
                    
                    if len(continent_totals) > 0:
                        # Select colors for this conference's continents
                        conf_colors = colors[:len(continent_totals)]
                        
                        wedges, texts, autotexts = axes[i].pie(
                            continent_totals.values, 
                            labels=continent_totals.index, 
                            autopct='%1.1f%%', 
                            colors=conf_colors, 
                            startangle=90
                        )
                        
                        axes[i].set_title(f'{conference}\n({continent_totals.sum()} papers)', 
                                        fontsize=12, fontweight='bold')
                        
                        # Make percentage text more readable
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('bold')
                            autotext.set_fontsize(9)
                    else:
                        axes[i].text(0.5, 0.5, f'{conference}\nNo data', 
                                   ha='center', va='center', transform=axes[i].transAxes,
                                   fontsize=12, fontweight='bold')
                        axes[i].set_xlim(0, 1)
                        axes[i].set_ylim(0, 1)
                
                # Hide unused subplots
                for i in range(n_conferences, len(axes)):
                    axes[i].set_visible(False)
                
                plt.suptitle('Distribution of Papers by Continent per Conference', 
                           fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout()
                plt.subplots_adjust(top=0.93)  # Make room for suptitle
                
                self.create_matplotlib_chart(fig, "Papers Distribution by Conference and Continent")
            else:
                # Fallback: single pie chart if no conference data
                fig, ax = plt.subplots(figsize=(10, 8))
                
                continent_totals = df_pandas.groupby('source_predominant_continent')['paper_count'].sum()
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(continent_totals)))
                wedges, texts, autotexts = ax.pie(continent_totals.values, labels=continent_totals.index, 
                                                autopct='%1.1f%%', colors=colors, startangle=90)
                
                ax.set_title('Distribution of Papers by Continent')
                
                # Make percentage text more readable
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                plt.tight_layout()
                self.create_matplotlib_chart(fig, "Papers Distribution by Continent")
        
        with tab4:
            if 'source_year' in df_pandas.columns and 'source_conference' in df_pandas.columns:
                fig, ax = plt.subplots(figsize=(14, 10))
                
                pivot_data = df_pandas.pivot_table(
                    values='paper_count',
                    index='source_conference',
                    columns='source_year',
                    aggfunc='sum',
                    fill_value=0
                )
                
                im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
                
                # Set ticks and labels
                ax.set_xticks(np.arange(len(pivot_data.columns)))
                ax.set_yticks(np.arange(len(pivot_data.index)))
                ax.set_xticklabels(pivot_data.columns)
                ax.set_yticklabels(pivot_data.index)
                
                # Rotate the tick labels and set their alignment
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Add colorbar
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel('Paper Count', rotation=-90, va="bottom")
                
                # Add text annotations
                for i in range(len(pivot_data.index)):
                    for j in range(len(pivot_data.columns)):
                        text = ax.text(j, i, int(pivot_data.iloc[i, j]),
                                     ha="center", va="center", color="black" if pivot_data.iloc[i, j] < pivot_data.values.max()/2 else "white")
                
                ax.set_title('Papers Heatmap: Conference vs Year')
                ax.set_xlabel('Year')
                ax.set_ylabel('Conference')
                
                plt.tight_layout()
                self.create_matplotlib_chart(fig, "Papers Heatmap")
    
    def create_committee_visualizations(self, df: pl.DataFrame):
        """Create matplotlib visualizations for committee data"""
        if df.height == 0:
            return
            
        df_pandas = df.to_pandas()
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Bar Chart", "📈 Line Chart", "🥧 Pie Chart", "🌍 Geographic"])
        
        with tab1:
            if 'conference' in df_pandas.columns:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                conference_data = df_pandas.groupby(['conference', 'continent'])['committee_count'].sum().reset_index()
                
                conferences = conference_data['conference'].unique()
                continents = conference_data['continent'].unique()
                
                x = np.arange(len(conferences))
                width = 0.8 / len(continents)
                
                for i, continent in enumerate(continents):
                    continent_data = conference_data[conference_data['continent'] == continent]
                    values = [continent_data[continent_data['conference'] == conf]['committee_count'].sum() 
                             if conf in continent_data['conference'].values else 0 
                             for conf in conferences]
                    
                    ax.bar(x + i * width, values, width, label=continent, alpha=0.8)
                
                ax.set_xlabel('Conference')
                ax.set_ylabel('Number of Committee Members')
                ax.set_title('Committee Members by Conference and Continent')
                ax.set_xticks(x + width * (len(continents) - 1) / 2)
                ax.set_xticklabels(conferences, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                self.create_matplotlib_chart(fig, "Committee Members by Conference")
        
        with tab2:
            if 'year' in df_pandas.columns:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                yearly_data = df_pandas.groupby(['year', 'continent'])['committee_count'].sum().reset_index()
                
                for continent in yearly_data['continent'].unique():
                    continent_data = yearly_data[yearly_data['continent'] == continent]
                    ax.plot(continent_data['year'], continent_data['committee_count'], 
                           marker='o', linewidth=2, markersize=6, label=continent)
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of Committee Members')
                ax.set_title('Committee Trends Over Time by Continent')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                self.create_matplotlib_chart(fig, "Committee Trends Over Time")
        
        with tab3:
            if 'conference' in df_pandas.columns:
                conferences = df_pandas['conference'].unique()
                
                # Calculate number of rows and columns for subplots
                n_conferences = len(conferences)
                n_cols = min(3, n_conferences)  # Max 3 columns
                n_rows = (n_conferences + n_cols - 1) // n_cols  # Ceiling division
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
                
                # Handle case where there's only one subplot
                if n_conferences == 1:
                    axes = [axes]
                elif n_rows == 1 and n_cols > 1:
                    axes = axes.flatten()
                elif n_rows > 1:
                    axes = axes.flatten()
                
                colors = plt.cm.Pastel1(np.linspace(0, 1, 10))  # Generate enough colors
                
                for i, conference in enumerate(conferences):
                    # Filter data for this conference
                    conf_data = df_pandas[df_pandas['conference'] == conference]
                    continent_totals = conf_data.groupby('continent')['committee_count'].sum()
                    
                    if len(continent_totals) > 0:
                        # Select colors for this conference's continents
                        conf_colors = colors[:len(continent_totals)]
                        
                        wedges, texts, autotexts = axes[i].pie(
                            continent_totals.values, 
                            labels=continent_totals.index, 
                            autopct='%1.1f%%', 
                            colors=conf_colors, 
                            startangle=90
                        )
                        
                        axes[i].set_title(f'{conference}\n({continent_totals.sum()} members)', 
                                        fontsize=12, fontweight='bold')
                        
                        # Make percentage text more readable
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('bold')
                            autotext.set_fontsize(9)
                    else:
                        axes[i].text(0.5, 0.5, f'{conference}\nNo data', 
                                   ha='center', va='center', transform=axes[i].transAxes,
                                   fontsize=12, fontweight='bold')
                        axes[i].set_xlim(0, 1)
                        axes[i].set_ylim(0, 1)
                
                # Hide unused subplots
                for i in range(n_conferences, len(axes)):
                    axes[i].set_visible(False)
                
                plt.suptitle('Distribution of Committee Members by Continent per Conference', 
                           fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout()
                plt.subplots_adjust(top=0.93)  # Make room for suptitle
                
                self.create_matplotlib_chart(fig, "Committee Distribution by Conference and Continent")
            else:
                # Fallback: single pie chart if no conference data
                fig, ax = plt.subplots(figsize=(10, 8))
                
                continent_totals = df_pandas.groupby('continent')['committee_count'].sum()
                
                colors = plt.cm.Pastel1(np.linspace(0, 1, len(continent_totals)))
                wedges, texts, autotexts = ax.pie(continent_totals.values, labels=continent_totals.index, 
                                                autopct='%1.1f%%', colors=colors, startangle=90)
                
                ax.set_title('Distribution of Committee Members by Continent')
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                plt.tight_layout()
                self.create_matplotlib_chart(fig, "Committee Distribution by Continent")
        
        with tab4:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            continent_totals = df_pandas.groupby('continent')['committee_count'].sum().reset_index()
            
            # Create a bubble chart
            x_pos = np.arange(len(continent_totals))
            sizes = continent_totals['committee_count'] * 50  # Scale for visibility
            
            scatter = ax.scatter(x_pos, continent_totals['committee_count'], 
                               s=sizes, alpha=0.6, c=range(len(continent_totals)), cmap='viridis')
            
            ax.set_xlabel('Continent')
            ax.set_ylabel('Number of Committee Members')
            ax.set_title('Committee Members Distribution by Continent')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(continent_totals['continent'], rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on points
            for i, (continent, count) in enumerate(zip(continent_totals['continent'], continent_totals['committee_count'])):
                ax.annotate(f'{count}', (i, count), xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            self.create_matplotlib_chart(fig, "Committee Geographic Distribution")
    
    def run_analytics_queries(self):
        """Run analytics queries based on selected analysis type"""
        if not st.session_state.analytics_client:
            st.error("Please initialize the database connection first!")
            return
        
        st.header("📊 Analytics Dashboard")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "Papers by Conference, Continent & Year",
                "Papers by Conference & Continent",
                "Committees by Conference, Country & Year", 
                "Committees by Continent & Year",
                "Committees by Continent"
            ],
            key="analysis_type_selector"
        )
        
        # Show description of selected analysis
        analysis_descriptions = {
            "Papers by Conference, Continent & Year": "📄 Analyze paper counts across conferences, continents, and years",
            "Papers by Conference & Continent": "📊 Compare paper distribution by conference and continent",
            "Committees by Conference, Country & Year": "👥 Track committee member distribution by conference, country, and year",
            "Committees by Continent & Year": "🌍 Analyze committee member trends across continents over time",
            "Committees by Continent": "🌐 Overview of committee member distribution by continent"
        }
        
        if analysis_type in analysis_descriptions:
            st.info(analysis_descriptions[analysis_type])
        
        # Get filters based on selected analysis type
        filters = self.render_filters_sidebar(analysis_type)
        
        if st.button("🚀 Run Analysis", type="primary"):
            try:
                with st.spinner("Running analysis..."):
                    if analysis_type == "Papers by Conference, Continent & Year":
                        result = st.session_state.analytics_client.query_paper_count_per_conference_continent_and_year(
                            conferences=filters['conferences'],
                            years=filters['years'],
                            continents=filters['continents']
                        )
                        self.display_dataframe_with_download(result, "Papers by Conference, Continent & Year", "papers_conf_cont_year")
                        self.create_paper_visualizations(result)
                        
                    elif analysis_type == "Papers by Conference & Continent":
                        result = st.session_state.analytics_client.query_paper_count_per_conference_and_continent(
                            conferences=filters['conferences'],
                            continents=filters['continents']
                        )
                        self.display_dataframe_with_download(result, "Papers by Conference & Continent", "papers_conf_cont")
                        self.create_paper_visualizations(result)
                        
                    elif analysis_type == "Committees by Conference, Country & Year":
                        result = st.session_state.analytics_client.get_committees_per_conference_country_year_count(
                            conferences=filters['conferences'],
                            years=filters['years']
                        )
                        self.display_dataframe_with_download(result, "Committees by Conference, Country & Year", "committees_conf_country_year")
                        
                    elif analysis_type == "Committees by Continent & Year":
                        result = st.session_state.analytics_client.get_committees_per_continent_year_count(
                            conferences=filters['conferences'],
                            continents=filters['continents'],
                            years=filters['years']
                        )
                        self.display_dataframe_with_download(result, "Committees by Continent & Year", "committees_cont_year")
                        self.create_committee_visualizations(result)
                        
                    elif analysis_type == "Committees by Continent":
                        result = st.session_state.analytics_client.get_committees_per_continent_count(
                            conferences=filters['conferences'],
                            continents=filters['continents']
                        )
                        self.display_dataframe_with_download(result, "Committees by Continent", "committees_cont")
                        self.create_committee_visualizations(result)
                        
            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")
                st.exception(e)
    
    def run(self):
        """Main application runner"""
        st.title("📊 Paper Analytics Dashboard")
        st.markdown("*Using matplotlib visualizations with multi-database integration*")
        st.markdown("---")
        
        # Setup connection
        self.setup_connection()
        
        # Show connection status
        if st.session_state.analytics_client:
            st.sidebar.success("🟢 Database Connected")
        else:
            st.sidebar.warning("🟡 Please initialize database connection")
        
        # Run analytics (filters are rendered inside based on analysis type)
        self.run_analytics_queries()
        
        # Footer
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit, Matplotlib & Seaborn")

# Run the application
if __name__ == "__main__":
    app = StreamlitPaperAnalytics()
    app.run()