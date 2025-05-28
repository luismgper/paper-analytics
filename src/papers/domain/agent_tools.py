import polaras as pl


class AgentTools:
    
    def __init__(self, df_citations: pl.DAtaframe):
        self.df_citations = df_citations
    
    def get_most_cited_countries(self, text) -> str:
        """Get top cited countries from the dataframe"""
        print("tool 1")
        df = (
            self.df_citations.filter(pl.col("cited_country") != "UNKNOWN")
            .group_by(["cited_country"])\
            .agg(pl.count().alias("citation_count"))\
            .sort("citation_count", descending=True)\
            .limit(10)
        )
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
        ):    
            return "Here are the most cited countries in markdown format:  \n" + str((df).head(10))
        
    def get_publications_per_year(self, text) -> str:
        """Get publication count by year with optional filtering"""
        print("tool 2")
        return "Don't have enough data"