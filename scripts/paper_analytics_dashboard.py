import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl
import numpy as np
import io
from matplotlib import patheffects

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
            "Papers by Conference and Continent": ["text", "conferences", "continents"],
            "Citations by Conference, Continent and Year": ["text","conferences", "years", "continents", "cited_continents"],
            "Citations by Conference and Continent": ["text", "conferences", "continents", "cited_continents"],            
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
                options=[str(year) for year in range(2012, 2023)],
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
            # col1, col2, col3 = st.columns(3)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", df.height)
            # with col2:
            #     st.metric("Total Columns", df.width)
            # with col3:
            with col2:
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
        if 'source_year' in df_pandas.columns:
            tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "📈 Line Chart", "🥧 Pie Chart"])
        else:
            tab1, tab3 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart"])
                
        with tab1:
            if 'source_conference' in df_pandas.columns:
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
                ax.set_ylabel('Number of papers')
                ax.set_title('Papers by conference and continent')
                ax.set_xticks(x + width * (len(continents) - 1) / 2)
                ax.set_xticklabels(conferences, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                self.create_matplotlib_chart(fig, "Papers by conference and continent")
        
        if 'source_year' in df_pandas.columns:
            with tab2:
                # fig, ax = plt.subplots(figsize=(14, 8))
                
                # yearly_data = df_pandas.groupby(['source_year', 'source_predominant_continent'])['paper_count'].sum().reset_index()
                
                # for continent in yearly_data['source_predominant_continent'].unique():
                #     continent_data = yearly_data[yearly_data['source_predominant_continent'] == continent]
                #     ax.plot(continent_data['source_year'], continent_data['paper_count'], 
                #            marker='o', linewidth=2, markersize=6, label=continent)
                
                # ax.set_xlabel('Year')
                # ax.set_ylabel('Number of Papers')
                # ax.set_title('Paper Trends Over Time by Continent')
                # ax.legend()
                # ax.grid(True, alpha=0.3)
                # plt.xticks(rotation=45)
                # plt.tight_layout()
                
                # self.create_matplotlib_chart(fig, "Paper Trends Over Time")
                
                # Get unique conferences
                conferences = df_pandas['source_conference'].unique()
                num_conferences = len(conferences)
                
                # Calculate subplot grid dimensions
                cols = min(2, num_conferences)  # Max 3 columns
                rows = (num_conferences + cols - 1) // cols  # Ceiling division
                
                # Create subplots
                fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
                
                # Handle case where there's only one subplot
                if num_conferences == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
                else:
                    axes = axes.flatten()
                
                # Plot each conference
                for i, conference in enumerate(conferences):
                    ax = axes[i]
                    
                    # Filter data for this conference
                    conference_data = df_pandas[df_pandas['source_conference'] == conference]
                    yearly_data = conference_data.groupby(['source_year', 'source_predominant_continent'])['paper_count'].sum().reset_index()
                    
                    # Plot each continent for this conference
                    for continent in yearly_data['source_predominant_continent'].unique():
                        continent_data = yearly_data[yearly_data['source_predominant_continent'] == continent]
                        ax.plot(continent_data['source_year'], continent_data['paper_count'],
                            marker='o', linewidth=2, markersize=6, label=continent)
                    
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Number of papers')
                    ax.set_title(f'Paper trends - {conference}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='x', rotation=45)
                
                # Hide empty subplots if any
                for i in range(num_conferences, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                self.create_matplotlib_chart(fig, "Paper trends over time")               
        
        with tab3:
            if 'source_conference' in df_pandas.columns:
                conferences = df_pandas['source_conference'].unique()
                
                # Calculate number of rows and columns for subplots
                n_conferences = len(conferences)
                n_cols = min(3, n_conferences)  # Max 3 columns
                n_rows = (n_conferences + n_cols - 1) // n_cols  # Ceiling division
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
                
                # Handle case where there's only one subplot
                if n_conferences == 1:
                    axes = [axes]
                elif n_rows == 1 and n_cols > 1:
                    axes = axes.flatten()
                elif n_rows > 1:
                    axes = axes.flatten()
                
                colors = plt.cm.Set3(np.linspace(0, 1, 10))
                
                for i, conference in enumerate(conferences):
                    # Filter data for this conference
                    conf_data = df_pandas[df_pandas['source_conference'] == conference]
                    continent_totals = conf_data.groupby('source_predominant_continent')['paper_count'].sum()
                    
                    if len(continent_totals) > 0:
                        # Calculate percentages for legend
                        total = continent_totals.sum()
                        percentages = [(value/total)*100 for value in continent_totals.values]
                        
                        # Select colors for this conference's continents
                        conf_colors = colors[:len(continent_totals)]
                        
                        # Create pie chart without percentage labels
                        wedges, texts = axes[i].pie(
                            continent_totals.values, 
                            labels=None,  # No labels on the pie
                            colors=conf_colors, 
                            startangle=90,
                            pctdistance=0.85,  
                            labeldistance=1.1,                             
                        )
                        
                        axes[i].set_title(f'{conference}\n({continent_totals.sum()} papers)', 
                                        fontsize=11, fontweight='bold')
                        
                        # Create legend with percentages
                        legend_labels = [f'{continent}: {percent:.1f}%' 
                                       for continent, percent in zip(continent_totals.index, percentages)]
                        
                        axes[i].legend(wedges, legend_labels, 
                                     title="Continents", 
                                     loc="center left", 
                                     bbox_to_anchor=(1, 0, 0.5, 1),
                                     fontsize=10)
                        
                    else:
                        axes[i].text(0.5, 0.5, f'{conference}\nNo data', 
                                   ha='center', va='center', transform=axes[i].transAxes,
                                   fontsize=12, fontweight='bold')
                        axes[i].set_xlim(0, 1)
                        axes[i].set_ylim(0, 1)
                
                # Hide unused subplots
                for i in range(n_conferences, len(axes)):
                    axes[i].set_visible(False)
                
                plt.suptitle('Distribution of papers by continent per conference', 
                           fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout()
                plt.subplots_adjust(top=0.93, right=0.85)  # Make room for legends
                
                self.create_matplotlib_chart(fig, "Papers distribution by conference and continent")    
        
        # with tab3:
        #     if 'source_conference' in df_pandas.columns:
        #         conferences = df_pandas['source_conference'].unique()
                
        #         # Calculate number of rows and columns for subplots
        #         n_conferences = len(conferences)
        #         n_cols = min(3, n_conferences)  # Max 3 columns
        #         n_rows = (n_conferences + n_cols - 1) // n_cols  # Ceiling division
                
        #         fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
                
        #         # Handle case where there's only one subplot
        #         if n_conferences == 1:
        #             axes = [axes]
        #         elif n_rows == 1 and n_cols > 1:
        #             axes = axes.flatten()
        #         elif n_rows > 1:
        #             axes = axes.flatten()
                
        #         colors = plt.cm.Set3(np.linspace(0, 1, 10))  # Generate enough colors
                
        #         for i, conference in enumerate(conferences):
        #             # Filter data for this conference
        #             conf_data = df_pandas[df_pandas['source_conference'] == conference]
        #             continent_totals = conf_data.groupby('source_predominant_continent')['paper_count'].sum()
                    
        #             if len(continent_totals) > 0:
        #                 # Select colors for this conference's continents
        #                 conf_colors = colors[:len(continent_totals)]
                        
        #                 wedges, texts, autotexts = axes[i].pie(
        #                     continent_totals.values, 
        #                     labels=continent_totals.index, 
        #                     autopct='%1.1f%%', 
        #                     colors=conf_colors, 
        #                     startangle=90
        #                 )
                        
        #                 axes[i].set_title(f'{conference}\n({continent_totals.sum()} papers)', 
        #                                 fontsize=12, fontweight='bold')
                        
        #                 # Make percentage text more readable
        #                 for autotext in autotexts:
        #                     autotext.set_color('black')
        #                     autotext.set_fontweight('bold')
        #                     autotext.set_fontsize(9)
        #             else:
        #                 axes[i].text(0.5, 0.5, f'{conference}\nNo data', 
        #                            ha='center', va='center', transform=axes[i].transAxes,
        #                            fontsize=12, fontweight='bold')
        #                 axes[i].set_xlim(0, 1)
        #                 axes[i].set_ylim(0, 1)
                
        #         # Hide unused subplots
        #         for i in range(n_conferences, len(axes)):
        #             axes[i].set_visible(False)
                
        #         plt.suptitle('Distribution of Papers by Continent per Conference', 
        #                    fontsize=16, fontweight='bold', y=0.98)
        #         plt.tight_layout()
        #         plt.subplots_adjust(top=0.93)  # Make room for suptitle
                
        #         self.create_matplotlib_chart(fig, "Papers Distribution by Conference and Continent")
        #     else:
        #         # Fallback: single pie chart if no conference data
        #         fig, ax = plt.subplots(figsize=(10, 8))
                
        #         continent_totals = df_pandas.groupby('source_predominant_continent')['paper_count'].sum()
                
        #         colors = plt.cm.Set3(np.linspace(0, 1, len(continent_totals)))
        #         wedges, texts, autotexts = ax.pie(continent_totals.values, labels=continent_totals.index, 
        #                                         autopct='%1.1f%%', colors=colors, startangle=90)
                
        #         ax.set_title('Distribution of Papers by Continent')
                
        #         # Make percentage text more readable
        #         for autotext in autotexts:
        #             autotext.set_color('white')
        #             autotext.set_fontweight('bold')
                
        #         plt.tight_layout()
        #         self.create_matplotlib_chart(fig, "Papers Distribution by Continent")
        
    def create_citation_visualizations(self, df: pl.DataFrame):
        """Create matplotlib visualizations for paper data"""
        if df.height == 0:
            return
            
        df_pandas = df.to_pandas()
        
        # Create tabs for different visualizations
        if 'source_year' in df_pandas.columns:
            tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "📈 Line Chart", "🥧 Pie Chart"])
        else:
            tab1, tab3 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart"])
                
        with tab1:
            print(df_pandas.columns)
            if 'source_conference' in df_pandas.columns:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                grouped_data = df_pandas.groupby(['source_conference', 'cited_predominant_continent'])['paper_count'].sum().reset_index()
                
                # Create grouped bar chart
                conferences = grouped_data['source_conference'].unique()
                continents = grouped_data['cited_predominant_continent'].unique()
                
                x = np.arange(len(conferences))
                width = 0.8 / len(continents)
                
                for i, continent in enumerate(continents):
                    continent_data = grouped_data[grouped_data['cited_predominant_continent'] == continent]
                    values = [continent_data[continent_data['source_conference'] == conf]['paper_count'].sum() 
                             if conf in continent_data['source_conference'].values else 0 
                             for conf in conferences]
                    
                    ax.bar(x + i * width, values, width, label=continent, alpha=0.8)
                
                ax.set_xlabel('Conference')
                ax.set_ylabel('Number of citations')
                ax.set_title('Citations by conference and continent')
                ax.set_xticks(x + width * (len(continents) - 1) / 2)
                ax.set_xticklabels(conferences, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                self.create_matplotlib_chart(fig, "Citations by conference and continent")
        
        if 'source_year' in df_pandas.columns:
            with tab2:                
                # Get unique conferences
                conferences = df_pandas['source_conference'].unique()
                num_conferences = len(conferences)
                
                # Calculate subplot grid dimensions
                cols = min(2, num_conferences)  # Max 3 columns
                rows = (num_conferences + cols - 1) // cols  # Ceiling division
                
                # Create subplots
                fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
                
                # Handle case where there's only one subplot
                if num_conferences == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
                else:
                    axes = axes.flatten()
                
                # Plot each conference
                for i, conference in enumerate(conferences):
                    ax = axes[i]
                    
                    # Filter data for this conference
                    conference_data = df_pandas[df_pandas['source_conference'] == conference]
                    yearly_data = conference_data.groupby(['source_year', 'cited_predominant_continent'])['paper_count'].sum().reset_index()
                    
                    # Plot each continent for this conference
                    for continent in yearly_data['cited_predominant_continent'].unique():
                        continent_data = yearly_data[yearly_data['cited_predominant_continent'] == continent]
                        ax.plot(continent_data['source_year'], continent_data['paper_count'],
                            marker='o', linewidth=2, markersize=6, label=continent)
                    
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Number of citations')
                    ax.set_title(f'Citation trends - {conference}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='x', rotation=45)
                
                # Hide empty subplots if any
                for i in range(num_conferences, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                self.create_matplotlib_chart(fig, "Citation trends over time")               
        
        with tab3:
            if 'source_conference' in df_pandas.columns:
                conferences = df_pandas['source_conference'].unique()
                
                # Calculate number of rows and columns for subplots
                n_conferences = len(conferences)
                n_cols = min(3, n_conferences)  # Max 3 columns
                n_rows = (n_conferences + n_cols - 1) // n_cols  # Ceiling division
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
                
                # Handle case where there's only one subplot
                if n_conferences == 1:
                    axes = [axes]
                elif n_rows == 1 and n_cols > 1:
                    axes = axes.flatten()
                elif n_rows > 1:
                    axes = axes.flatten()
                
                colors = plt.cm.Set3(np.linspace(0, 1, 10))
                
                for i, conference in enumerate(conferences):
                    # Filter data for this conference
                    conf_data = df_pandas[df_pandas['source_conference'] == conference]
                    continent_totals = conf_data.groupby('cited_predominant_continent')['paper_count'].sum()
                    
                    if len(continent_totals) > 0:
                        # Calculate percentages for legend
                        total = continent_totals.sum()
                        percentages = [(value/total)*100 for value in continent_totals.values]
                        
                        # Select colors for this conference's continents
                        conf_colors = colors[:len(continent_totals)]
                        
                        # Create pie chart without percentage labels
                        wedges, texts = axes[i].pie(
                            continent_totals.values, 
                            labels=None,  # No labels on the pie
                            colors=conf_colors, 
                            startangle=90,
                            pctdistance=0.85,  
                            labeldistance=1.1,                             
                        )
                        
                        axes[i].set_title(f'{conference}\n({continent_totals.sum()} papers)', 
                                        fontsize=11, fontweight='bold')
                        
                        # Create legend with percentages
                        legend_labels = [f'{continent}: {percent:.1f}%' 
                                       for continent, percent in zip(continent_totals.index, percentages)]
                        
                        axes[i].legend(wedges, legend_labels, 
                                     title="Continents", 
                                     loc="center left", 
                                     bbox_to_anchor=(1, 0, 0.5, 1),
                                     fontsize=10)
                        
                    else:
                        axes[i].text(0.5, 0.5, f'{conference}\nNo data', 
                                   ha='center', va='center', transform=axes[i].transAxes,
                                   fontsize=12, fontweight='bold')
                        axes[i].set_xlim(0, 1)
                        axes[i].set_ylim(0, 1)
                
                # Hide unused subplots
                for i in range(n_conferences, len(axes)):
                    axes[i].set_visible(False)
                
                plt.suptitle('Distribution of citations by continent per conference', 
                           fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout()
                plt.subplots_adjust(top=0.93, right=0.85)  # Make room for legends
                
                self.create_matplotlib_chart(fig, "Citation distribution by conference and continent")    
                
    def create_citation_by_source_and_cited_continent_visualizations(self, df: pl.DataFrame):
        """Create Sankey diagram visualizations for citation data"""
        if df.height == 0:
            return
            
        df_pandas = df.to_pandas()
        
        # Create single tab for Sankey diagram
        tab1 = st.tabs(["🔄 Citation Flow Sankey"])
        
        with tab1[0]:
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
                    height=400 * rows,  # Adjust height based on number of rows
                    margin=dict(l=50, r=50, t=80, b=50),
                    showlegend=False
                )
                
                # Display the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Required columns not found in dataframe. Expected: 'source_conference', 'source_predominant_continent', 'cited_predominant_continent', 'paper_count'")
    
    def create_committee_visualizations(self, df: pl.DataFrame):
        """Create matplotlib visualizations for committee data"""
        if df.height == 0:
            return
        
        print("aqui 1")
            
        df_pandas = df.to_pandas()
        
        # Create tabs for different visualizations
        if 'year' in df_pandas.columns:
            tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "📈 Line Chart", "🥧 Pie Chart"])
        else:
            tab1, tab3 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart"])
        
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
                ax.set_ylabel('Number of committee members')
                ax.set_title('Committee members by conference and continent')
                ax.set_xticks(x + width * (len(continents) - 1) / 2)
                ax.set_xticklabels(conferences, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                self.create_matplotlib_chart(fig, "Committee members by conference")
        
        if 'year' in df_pandas.columns:
            with tab2:
                if 'conference' in df_pandas.columns and 'year' in df_pandas.columns:
                    # Get unique conferences
                    conferences = df_pandas['conference'].unique()
                    num_conferences = len(conferences)
                    
                    # Calculate subplot grid dimensions
                    cols = min(2, num_conferences)  # Max 3 columns
                    rows = (num_conferences + cols - 1) // cols  # Ceiling division
                    
                    # Create subplots
                    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
                    
                    # Handle case where there's only one subplot
                    if num_conferences == 1:
                        axes = [axes]
                    elif rows == 1:
                        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
                    else:
                        axes = axes.flatten()
                    
                    # Plot each conference
                    for i, conference in enumerate(conferences):
                        ax = axes[i]
                        
                        # Filter data for this conference
                        conference_data = df_pandas[df_pandas['conference'] == conference]
                        yearly_data = conference_data.groupby(['year', 'continent'])['committee_count'].sum().reset_index()
                        
                        # Plot each continent for this conference
                        for continent in yearly_data['continent'].unique():
                            continent_data = yearly_data[yearly_data['continent'] == continent]
                            ax.plot(continent_data['year'], continent_data['committee_count'],
                                marker='o', linewidth=2, markersize=6, label=continent)
                        
                        ax.set_xlabel('Year')
                        ax.set_ylabel('Number of committee members')
                        ax.set_title(f'Committee trends - {conference}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(axis='x', rotation=45)
                    
                    # Hide empty subplots if any
                    for i in range(num_conferences, len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    self.create_matplotlib_chart(fig, "Committee trends by conference")
                    
        with tab3:
            if 'conference' in df_pandas.columns:
                conferences = df_pandas['conference'].unique()
                
                # Calculate number of rows and columns for subplots
                n_conferences = len(conferences)
                n_cols = min(3, n_conferences)  # Max 3 columns
                n_rows = (n_conferences + n_cols - 1) // n_cols  # Ceiling division
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
                
                # Handle case where there's only one subplot
                if n_conferences == 1:
                    axes = [axes]
                elif n_rows == 1 and n_cols > 1:
                    axes = axes.flatten()
                elif n_rows > 1:
                    axes = axes.flatten()
                
                colors = plt.cm.Set3(np.linspace(0, 1, 10))
                
                for i, conference in enumerate(conferences):
                    # Filter data for this conference
                    conf_data = df_pandas[df_pandas['conference'] == conference]
                    continent_totals = conf_data.groupby('continent')['committee_count'].sum()
                    
                    if len(continent_totals) > 0:
                        # Calculate percentages for legend
                        total = continent_totals.sum()
                        percentages = [(value/total)*100 for value in continent_totals.values]
                        
                        # Select colors for this conference's continents
                        conf_colors = colors[:len(continent_totals)]
                        
                        # Create pie chart without percentage labels
                        wedges, texts = axes[i].pie(
                            continent_totals.values, 
                            labels=None,  # No labels on the pie
                            colors=conf_colors, 
                            startangle=90,
                            pctdistance=0.85,  
                            labeldistance=1.1,                             
                        )
                        
                        axes[i].set_title(f'{conference}\n({continent_totals.sum()} committees)', 
                                        fontsize=11, fontweight='bold')
                        
                        # Create legend with percentages
                        legend_labels = [f'{continent}: {percent:.1f}%' 
                                       for continent, percent in zip(continent_totals.index, percentages)]
                        
                        axes[i].legend(wedges, legend_labels, 
                                     title="Continents", 
                                     loc="center left", 
                                     bbox_to_anchor=(1, 0, 0.5, 1),
                                     fontsize=10)
                        
                    else:
                        axes[i].text(0.5, 0.5, f'{conference}\nNo data', 
                                   ha='center', va='center', transform=axes[i].transAxes,
                                   fontsize=12, fontweight='bold')
                        axes[i].set_xlim(0, 1)
                        axes[i].set_ylim(0, 1)
                
                # Hide unused subplots
                for i in range(n_conferences, len(axes)):
                    axes[i].set_visible(False)
                
                plt.suptitle('Distribution of committee members by continent', 
                           fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout()
                plt.subplots_adjust(top=0.93, right=0.85)  # Make room for legends
                
                self.create_matplotlib_chart(fig, "Committee distribution by conference and continent")    
    
    def create_committee_country_visualizations(self, df: pl.DataFrame):
        """Create matplotlib visualizations for committee country data"""
        if df.height == 0:
            return
            
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
                "Papers by Conference and Continent",
                "Citations by Conference, Continent and Year",
                "Citations by Conference and Continent",                
                "Citations by Conference and Source and Cited Continent",                
                "Committees by Conference, Country and Year", 
                "Committees by Continent and Year",
                "Committees by Continent"
            ],
            key="analysis_type_selector"
        )
        
        # Show description of selected analysis
        analysis_descriptions = {
            "Papers by Conference, Continent and Year": "Analyze paper counts across conferences, continents and years",
            "Papers by Conference and Continent": "Compare paper distribution by conference and continent",
            "Citations by Conference, Continent and Year": "Analyze citation counts across conferences, continents and years",
            "Citations by Conference and Continent": "Compare citation distribution by conference and continent",            
            "Citations by Conference and Source and Cited Continent": "Compare citation distribution by conference and source and cited continent",            
            "Committees by Conference, Country and Year": "Track committee member distribution by conference, country, and year",
            "Committees by Continent and Year": "Analyze committee member trends across continents over time",
            "Committees by Continent": "Overview of committee member distribution by continent"            
        }
        
        if analysis_type in analysis_descriptions:
            st.info(analysis_descriptions[analysis_type])
        
        # Get filters based on selected analysis type
        filters = self.render_filters_sidebar(analysis_type)
        
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
                        self.create_paper_visualizations(result)
                        
                    elif analysis_type == "Papers by Conference and Continent":
                        result = st.session_state.analytics_client.query_paper_count_per_conference_and_continent(
                            text=filters['text'],                            
                            conferences=filters['conferences'],
                            continents=filters['continents']
                        )
                        self.display_dataframe_with_download(result, "Papers by Conference and Continent", "papers_conf_cont")
                        self.create_paper_visualizations(result)
                        
                    if analysis_type == "Citations by Conference, Continent and Year":
                        result = st.session_state.analytics_client.query_citation_count_per_conference_continent_and_year(
                            text=filters['text'],
                            conferences=filters['conferences'],
                            years=filters['years'],
                            continents=filters['continents'],
                            cited_continents=filters['cited_continents'],
                        )
                        self.display_dataframe_with_download(result, "Papers by Conference, Continent and Year", "papers_conf_cont_year")
                        self.create_citation_visualizations(result)
                        
                    elif analysis_type == "Citations by Conference and Continent":
                        result = st.session_state.analytics_client.query_citation_count_per_conference_and_continent(
                            text=filters['text'],                            
                            conferences=filters['conferences'],
                            continents=filters['continents'],
                            cited_continents=filters['cited_continents'],
                        )
                        self.display_dataframe_with_download(result, "Papers by Conference and Continent", "papers_conf_cont")
                        self.create_citation_visualizations(result)                        
                        
                    elif analysis_type == "Citations by Conference and Source and Cited Continent":
                        result = st.session_state.analytics_client.query_citation_count_per_conference_source_continent_and_year(
                            text=filters['text'],                            
                            conferences=filters['conferences'],
                            continents=filters['continents'],
                            cited_continents=filters['cited_continents'],
                        )
                        self.display_dataframe_with_download(result, "Papers by Conference and Continent", "papers_conf_cont")
                        self.create_citation_by_source_and_cited_continent_visualizations(result)                        
                                                
                        
                    elif analysis_type == "Committees by Conference, Country and Year":
                        result = st.session_state.analytics_client.get_committees_per_conference_country_year_count(
                            conferences=filters['conferences'],
                            years=filters['years']
                        )
                        self.display_dataframe_with_download(result, "Committees by Conference, Country and Year", "committees_conf_country_year")
                        self.create_committee_country_visualizations(result)
                        
                    elif analysis_type == "Committees by Continent and Year":
                        result = st.session_state.analytics_client.get_committees_per_continent_year_count(
                            conferences=filters['conferences'],
                            continents=filters['continents'],
                            years=filters['years']
                        )
                        self.display_dataframe_with_download(result, "Committees by Continent and Year", "committees_cont_year")
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
        # st.title("📊 Paper Analytics Dashboard")
        st.title("Paper Analytics Dashboard")
        # st.markdown("*Using matplotlib visualizations with multi-database integration*")
        st.markdown("---")
        
        # Setup connection status display
        self.setup_connection()
        
        # Run analytics (filters are rendered inside based on analysis type)
        self.run_analytics_queries()
        
        # Footer
        # st.markdown("---")
        # st.markdown("Built with ❤️ using Streamlit, Matplotlib & Seaborn")

# Run the application
if __name__ == "__main__":
    app = StreamlitPaperAnalytics()
    app.run()