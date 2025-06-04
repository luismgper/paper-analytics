import polars as pl
import streamlit as st

class AgentTools:
    
    def __init__(self, df_citations: pl.DataFrame):
        self.df_citations = df_citations
        st.dataframe(df_citations, use_container_width=False)
        print("\nResults \n\n")
        print(f"Columns in dataframe: {df_citations.columns}")
        # with pl.Config(
        #     tbl_formatting="MARKDOWN",
        #     tbl_hide_column_data_types=True,
        #     tbl_hide_dataframe_shape=True,
        #     tbl_rows=-1,
        #     tbl_cols=-1,
        #     fmt_str_lengths=999999,
        # ):    
        #     print(str((df_citations).head(9999)))
            
    def get_other_citation_requested(self, text: str) -> str:
        """Method for when no other tool is able to provide data. For example, if only the topic is requested, a summary, non aggregation data, ..."""
        print("Other citation requested tool used:  " + text)
        df_simplified = self.df_citations.select(
            "source_title",
            "source_year",
            "source_conference",
            "source_predominant_country",
            "source_authors",
            "source_institutions",
            "source_summary",
        ).unique()
        
        
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_rows=-1,
            tbl_cols=-1,
            fmt_str_lengths=999999,
        ):    
            # print(str((df_simplified).head(9999)))
            return f"""Respond the user query {text} using the follwing paper data:  \n {str((df_simplified).head(9999))}"""# \n /no_think"""
    
    def get_most_cited_papers(self, text) -> str:
        """Get top cited papers from the dataframe"""
        print("citations per country tool")

        df = self.__get_dynamic_count_by_column(["cited_title", "cited_year"], descending=True)
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_rows=-1
        ):    
            return "Here are the most cited papers in markdown format:  \n" + str((df).head(9999)) 
    
    def get_most_cited_countries(self, text) -> str:
        """Get top cited countries from the dataframe"""
        print("citations per country tool")

        df = self.__get_dynamic_count_by_column(["cited_country"], descending=True)
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_rows=-1
        ):    
            return "Here are the most cited countries in markdown format:  \n" + str((df).head(9999))
        
    def get_publications_per_year(self, text) -> str:
        """Get publication count by year with optional filtering"""
        print("citations per year tool")

        df = self.__get_dynamic_count_by_column(["cited_year"], descending=True)
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_rows=-1
        ):    
            return  "Here are the citations per year in markdown format:  \n" + str((df).head(9999))
        
    def get_citations_per_institution(self, text) -> str:
        """Get citations count per institution"""
        print("citations per institution tool")

        # df = self.__get_dynamic_count_by_column(["cited_institution"], descending=True)
        df = self.df_citations.filter(pl.col(["cited_institution"]) != "UNKNOWN")\
            .group_by(["cited_institution"])\
            .agg(pl.len().alias("citation_count"))\
            .sort("citation_count", descending=False)
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_rows=-1
        ):    
            return  "Here are the citations per institution in markdown format:  \n" + str((df).head(9999))
    
    def __get_dynamic_count_by_column(self, columns: list[str], descending: bool = False) -> pl.DataFrame:
        """Reusable method to make count aggregations per indicated columns"""    
        df = self.df_citations
        # print(df)
        for column in columns:
            df = df.filter(pl.col(column) != "UNKNOWN")
        
        return (
            df.group_by(columns)
            .agg(pl.len().alias("citation_count"))
            .sort("citation_count", descending=descending)
        )