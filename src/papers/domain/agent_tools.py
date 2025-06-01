import polars as pl


class AgentTools:
    
    def __init__(self, df_citations: pl.DataFrame):
        self.df_citations = df_citations

    def get_other_citation_requested(self, text: str) -> str:
        """Method for when no other tool is able to provide data. For example, if only the topic is requested, a summary, non aggregation data, ..."""
        print("Other citation requested tool used")
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_rows=-1
        ):    
            return "Use the follwing paper data:  \n" + str((self.df_citations).head(9999999))
    
    def get_most_cited_papers(self, text) -> str:
        """Get top cited papers from the dataframe"""
        print("citations per country tool")

        df = self.__get_dynamic_count_by_column(["cited_title", "cited_year"], descending=True)
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
        ):    
            return "Here are the most cited papers in markdown format:  \n" + str((df).head(10)) 
    
    def get_most_cited_countries(self, text) -> str:
        """Get top cited countries from the dataframe"""
        print("citations per country tool")

        df = self.__get_dynamic_count_by_column(["cited_country"], descending=True)
        print(df)
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
        ):    
            return "Here are the most cited countries in markdown format:  \n" + str((df).head(10))
        
    def get_publications_per_year(self, text) -> str:
        """Get publication count by year with optional filtering"""
        print("citations per year tool")

        df = self.__get_dynamic_count_by_column(["cited_year"], descending=True)
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
        ):    
            return  "Here are the citations per year in markdown format:  \n" + str((df).head(10))
        
    def get_citations_per_institution(self, text) -> str:
        """Get citations count per institution"""
        print("citations per institution tool")

        df = self.__get_dynamic_count_by_column(["cited_institution"], descending=True)
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
        ):    
            return  "Here are the citations per institution in markdown format:  \n" + str((df).head(10))
    
    def __get_dynamic_count_by_column(self, columns: list[str], descending: bool = False) -> pl.DataFrame:
        """Reusable method to make count aggregations per indicated columns"""    
        df = self.df_citations
        print(df)
        for column in columns:
            df = df.filter(pl.col(column) != "UNKNOWN")
        
        return (
            df.group_by(columns)
            .agg(pl.len().alias("citation_count"))
            .sort("citation_count", descending=descending)
        )