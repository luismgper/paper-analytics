from src.papers.io import db
from src.papers.utils import models
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import polars as pl

class LogicConnector(str, Enum):
    none = ""
    and_op = "and"
    or_op = "or"

class ComparisonOperator(str, Enum):
    eq = "eq"
    lt = "lt"
    gt = "gt"
    le = "le"
    ge = "ge"
    ne = "ne"
    
class FilterCondition(BaseModel):
    level: int = 0
    connector: LogicConnector = LogicConnector.none
    field: str
    operator: ComparisonOperator
    values: list[str]    
    
class SourceFilters(BaseModel):
    text: Optional[str]
    filters: Optional[list[FilterCondition]]
    
class CitationFilters(BaseModel):
    text: Optional[str]
    filters: Optional[list[FilterCondition]]
    
class QueryParameters(BaseModel):
    source_filters: Optional[SourceFilters] = None
    citation_filters: Optional[CitationFilters] = None
    aggregations: Optional[dict] = None
    
class CitationParameters(BaseModel):
    title: str
    year: str
    
class MultiModalPaperQuery():
    """
    Class managing multimodal query methods. Allows querying of relational DB, vector DB and graph DB
    """
    
    GRAPH_DB_DATABASES = [
        "ccgrid",
        "socc",
        "europar",
        "eurosys",
        "ic2e",
        "icdcs",
        "ieeecloud",
        "middleware",
        "nsdi",
        "sigcomm",    
    ]
    
    VECTOR_DB = {
        "OUTPUT_FIELDS": [
            "Title",
            "TLDR",
            "Authors",
            "Institutions",
            "Countries",
            "Abstract",
            "KeyConcepts",
            "Year",
            "Conference",
            "Summary"
        ],        
        "SEARCH_FIELDS": [
            "AbstractVector", 
            "TitleVector", 
            "TLDRVector",
            "KeyConceptsVector"
        ],
        "LIST_FIELDS": [
            "Authors",
            "Institutions",
            "Countries",
        ],
        "EXPR_MAPPING": {
            ComparisonOperator.eq: "in",
            ComparisonOperator.lt: "<",
            ComparisonOperator.gt: ">",
            ComparisonOperator.ne: "not in",
            ComparisonOperator.le: "<=",
            ComparisonOperator.ge: ">=",
        }
    }
            
    def __init__(self, relational_db_client: db.SQLite, vector_db_client: db.Milvus, graph_db_client: db.Neo4j):
        self.relational_db_client = relational_db_client
        self.vector_db_client = vector_db_client
        self.graph_db_client = graph_db_client
    
    def query(self, query: QueryParameters) -> pl.DataFrame:
        
        # If source is provided, start by recovering source paper data.
        df_source = self.__get_data_from_source(query.source_filters) if query.source_filters else []

        # If citations are provided, make query with its filters
        df_target = self.__get_data_from_citations(query.target_filters) if query.citation_filters.filters else []
            
        return df_source, df_target
            
        # Merge results
        # df_merged = 
        
        # Apply aggregations over results
        
        
              
        
    def query_vector_db(self, text: str, expr: str = "", top_k: int = 10, search_fields: list = VECTOR_DB["SEARCH_FIELDS"]) -> pl.DataFrame:
        """Retrieve data from vector database

        Args:
            text (str): text for similarity search
            expr (str, optional): expression to filter metadata. Defaults to "".
            top_k (int, optional): number of entries to retrieve. Defaults to 10.

        Returns:
            pl.DataFrame: Results in polars dataframe format
        """
        
        # Vector DB query
        results = self.vector_db_client.search(
            text=text,
            output_fields=self.VECTOR_DB["OUTPUT_FIELDS"],
            limit=top_k,
            hybrid=True,
            hybrid_fields=search_fields,
            expr=expr,
        )
        
        # Create dataframe from results
        df_result = pl.DataFrame([result["entity"] for result in results])
        return df_result
        
    def get_citations(self, parameters: CitationParameters) -> pl.DataFrame:
        """Retrieve citations for a paper
        Args:
            parameters (CitationParameters): Needed parameters for query related to the paper. Containing title and year

        Returns:
            pl.DataFrame: DataFrame with query results
        """
        
        # Cypher query to search cited papers from source paper
        QUERY = """MATCH (p:Paper {title: $title, year: $year})
        OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
        OPTIONAL MATCH (cited)-[:HAS_INSTITUTION]->(inst:Institution)-[:LOCATED_IN]->(country:Country)
        RETURN p.title as paper_title, p.year as paper_year, p.PredominantCountry as predominant_country, p.Authors as authors,
        p.conference as conference, 
        cited.title AS cited_title, cited.PredominantCountry as cited_predominant_country, cited.conference AS cited_conference, 
        cited.Authors as cited_authors, country.name AS cited_country, inst.name as cited_institution
        """
        df_results = self.query_graph_db(query=QUERY, parameters=parameters.model_dump())
        return df_results
    
    def get_citers(self, parameters: CitationParameters) -> pl.DataFrame:
        """Retrieve papers citing a paper
        Args:
            parameters (CitationParameters): Needed parameters for query related to the paper. Containing title and year

        Returns:
            pl.DataFrame: DataFrame with query results
        """
        
        # Cypher query to search cited papers from source paper
        QUERY = """MATCH (cited:Paper {title: $title, year: $year})
        OPTIONAL MATCH (p:Paper)-[:CITES]->(cited)
        OPTIONAL MATCH (cited)-[:HAS_INSTITUTION]->(inst:Institution)-[:LOCATED_IN]->(country:Country)
        OPTIONAL MATCH (p)-[:HAS_INSTITUTION]->(p_inst:Institution)-[:LOCATED_IN]->(p_country:Country)
        RETURN p.title as paper_title, p.year as paper_year, p.PredominantCountry as predominant_country, p.Authors as authors,
        p.conference as conference, p_inst.name as source_institution, p_country.name as source_country,
        cited.title AS cited_title, cited.PredominantCountry as cited_predominant_country, cited.conference AS cited_conference, 
        cited.Authors as cited_authors, country.name AS cited_country, inst.name as cited_institution
        """
        df_results = self.query_graph_db(query=QUERY, parameters=parameters.model_dump())
        return df_results    
        
    def query_graph_db(self, query: str, parameters: dict = {}, databases: list = GRAPH_DB_DATABASES) -> pl.DataFrame:
        """General method to query graph database

        Args:
            query (str): Cypher query to be done
            parameters (dict, optional): parameters, if needed to be applied in Cypher query. Defaults to {}.
            databases (list, optional): List of databases to applyue the query to. Defaults to GRAPH_DB_DATABASES.

        Returns:
            pl.DataFrame: DataFrame with query results
        """
        
        # Recovers the query from each indicated database them in a dataframe
        results = []
        for database in databases:
            result = self.graph_db_client.run_query(
                query=query,
                database=database,
                parameters=parameters,
            )
            self.graph_db_client.close()
            
            for match in result:
                results.append({ "database": database, "match": match })
        
        df_exploded = pl.DataFrame(results).unnest("match")
        return df_exploded         
    
    def __get_citations_data_from_source(self, df_source: pl.DataFrame) -> pl.DataFrame:
        
        # UDF to use
        def get_citations_call(x): 
            print(x)
            return self.get_citations(
                CitationParameters(
                    **{"title": x["Title"], "year": x["Year"]}
                )
            ).to_dicts()
                
        df_papers_with_citations = (
            df_source.select(
                pl.all(),
                pl.struct("Title", "Year").alias("citers"),
            ).with_columns(
                pl.col("citers").map_elements(get_citations_call).alias("citers")
            )   
        )
        return df_papers_with_citations 
    
    def __get_citing_data_from_target(self, df_target: pl.DataFrame) -> pl.DataFrame:
        
        # UDF to use
        def get_citations_call(x): 
            print(x)
            return self.get_citers(
                CitationParameters(
                    **{"title": x["Title"], "year": x["Year"]}
                )
            ).to_dicts()
                
        df_papers_with_citations = (
            df_target.select(
                pl.all(),
                pl.struct("Title", "Year").alias("citations"),
            ).with_columns(
                pl.col("citations").map_elements(get_citations_call).alias("citations")
            )   
        )
        return df_papers_with_citations         
    
    def __get_data_from_source(self, filters: SourceFilters) -> pl.DataFrame:
        
        expr = self.__translate_vector_db_filters(filters)
        df_source_papers = self.query_vector_db(
            text=filters.text,
            expr=expr,
            # top_k=100
        )
        
        if len(df_source_papers) > 0:
            
            # Source data was recovered, then get its citation data
            df_source_with_citations = self.__get_citations_data_from_source(df_source_papers)      
            return df_source_with_citations
        
        return []
    
    def __get_data_from_citations(self, filters: CitationFilters) -> pl.DataFrame:
        expr = self.__translate_vector_db_filters(filters.filters)
        df_papers = self.query_vector_db(
            text=filters.text,
            expr=expr,
            search_fields=["TitleVector"]
            # top_k=100
        )

        if len(df_papers) > 0:
            
            # Papers found, cet citing papers
            df_target_with_citers = self.__get_citing_data_from_target(df_papers)
    
    
    def __translate_vector_db_filters(self, filter: SourceFilters) -> str:
        
        expr = ""
        for filter in filter.filters:
            # Translate filter condition to string
            expr_condition = self.__translate_vector_db_filter_condition(filter)
            
            # Concatenation o conditions depending on connector
            if filter.connector == LogicConnector.none:
                expr = expr_condition
            else:
                expr += f" {filter.condition} {expr_condition}"
                
        return expr
    
    def __translate_vector_db_filter_condition( self, filter: FilterCondition):
        if filter.field in self.VECTOR_DB["LIST_FIELDS"]:
            pass
        else:
            if filter.operator in [ComparisonOperator.ge, ComparisonOperator.le, ComparisonOperator.gt, ComparisonOperator.lt] and len(filter.values) != 1:
                raise TypeError(f"Only a single item is allowed as values for inequality operator in field {filter.field}")
            
            op = self.VECTOR_DB["EXPR_MAPPING"][filter.operator]
            return f"{filter.field} {op} {filter.values}"
    
    def __translate_graph_db_filters(self, ) -> str:
        pass
    
    def __translate_realtional_db_filters(self, ) -> str:
        pass