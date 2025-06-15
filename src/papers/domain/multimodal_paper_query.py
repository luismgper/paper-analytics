from src.papers.io import db
from src.papers.utils import models
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import polars as pl

class LogicConnector(str, Enum):
    """
    Logic connector to join two filter conditions
    """
    none = ""
    and_op = "and"
    or_op = "or"

class ComparisonOperator(str, Enum):
    """
    Valid comparison operators to be used in filtering
    """
    eq = "eq"
    lt = "lt"
    gt = "gt"
    le = "le"
    ge = "ge"
    ne = "ne"
    
class ParenthesisIndicator(str, Enum):
    """
    Indicates opening or closing of a nested condition
    """
    # none = ""
    open = "("
    close = ")" 
    
class FilterFields(str, Enum):
    """
    Fields to use as filtering
    """
    title="Title"
    year="Year"
    authors="Authors"
    institutions="Institutions"
    countries="Countries"
    committees="Committees"
    
class OutputFields(str, Enum):
    """
    Fields to be used as output to display results
    """
    source_title="Title"
    source_year="Year"
    source_authors="Authors"
    source_institutions="Institutions"
    source_countries="Countries"
    source_committees="Committees"
    citation_title="Title"
    citation_year="Year"
    citation_authors="Authors"
    citation_institutions="Institutions"
    citation_countries="Countries"
    citation_committees="Committees"     
    
class AggregationOperations(str, Enum):
    """
    Possible aggregation operations
    """
    count = "count"
    mean = "mean"
    
class SortOptions(BaseModel):
    """
    Possible sorting options in output
    """
    field: str = Field(description="Field to sort")
    descending: Optional[bool] = Field(default=True, description="Indicates descending ordering")   
    
class OutputParameters(BaseModel):
    """
    Output options to display results of the query
    """
    fields: list[OutputFields] = Field(default="Fields to return in output results")
    distinct: Optional[bool] = Field(default=False, description="Indicates if returning duplicate entries should be avoided")
            
class FilterCondition(BaseModel):
    """
    Provides individual fitlering condition for queries
    """
    parenthesis : Optional[ParenthesisIndicator] = Field(
        default="", 
        description="""Indicates if the opening or closing of a parenthesis grouping of conditions is needed. 
        If open is indication, the condition is the first of the grouped statement. 
        If close, it is the last of an already open statement""")
    connector: Optional[LogicConnector] = Field(
        default=LogicConnector.none, 
        description="Indicates if logical connector is applied before the condition")
    field: FilterFields = Field(
        description="Field of the filter condition")
    operator: ComparisonOperator = Field(
        description="Comparison operator of the filtering condition")
    values: list[str] = Field(
        default="List of values to use in the filter condition")
    
class SourceFilters(BaseModel):
    """
    Provides filtering criteria over source papers (papers that cite)
    """
    text: Optional[str] = Field(description="Topic regarding paper content of the query being made")
    filters: Optional[list[FilterCondition]] = Field(default=[], description="Filtering conditions about the papers")
    
class CitationFilters(BaseModel):
    """
    Provides filtering criteria over papers being cited
    """    
    text: Optional[str] = Field(description="Topic regarding cited paper of the query being made")
    filters: Optional[list[FilterCondition]] = Field(default=[], description="Filtering conditions about the papers")

class AggregationParameters(BaseModel):
    """
    Parameters to provide aggregations over retrieved data
    """
    group_by: Optional[list[OutputFields]] = Field(default=[], description="Fields to do grouping with")
    aggregations: Optional[list[AggregationOperations]] = Field(default=None, description="Aggregation operations requested")
    sort: Optional[list[SortOptions]] = Field(default=None, description="Sorting fields")    
    limit: int = Field(default=10, description="Number of recrods expected to return")
    
class QueryParameters(BaseModel):
    """
    Paper query options
    """
    source_filters: Optional[SourceFilters] = Field(default=None, description="Filter options over source papers")
    citation_filters: Optional[CitationFilters] = Field(default=None, description="Filter options over papers cited")
    aggregations: Optional[AggregationParameters] = Field(default=None, description="Aggregations to be made over results")
    output: Optional[OutputParameters] = Field(default=None, description="Options over returned results")
    
class CitationParameters(BaseModel):
    """
    Parameters to be used to filter graph DB
    """
    title: str = Field(description="Paper title")
    year: str = Field(description="Paper pubblishing year")
    
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
    
    RELATIONAL_DB =  {
        "CONFERENCE_MAPPING": {
            "ccgrid": "data/RelationalDB/ccgrid.db",
            "cloud": "data/RelationalDB/cloud.db",
            "europar": "data/RelationalDB/europar.db",
            "eurosys": "data/RelationalDB/eurosys.db",
            "ic2e": "data/RelationalDB/ic2e.db",
            "icdcs": "data/RelationalDB/icdcs.db",
            "IEEEcloud": "data/RelationalDB/IEEEcloud.db",
            "middleware": "data/RelationalDB/middleware.db",
            "nsdi": "data/RelationalDB/nsdi.db",
            "sigcomm": "data/RelationalDB/sigcomm.db",
        }
    }
            
    def __init__(self, relational_db_client: db.SQLite, vector_db_client: db.Milvus, graph_db_client: db.Neo4j):
        self.relational_db_client = relational_db_client
        self.vector_db_client = vector_db_client
        self.graph_db_client = graph_db_client
    
    def query(self, query: QueryParameters) -> pl.DataFrame:
        print(query)
        
        # If source is provided, start by recovering source paper data.
        df_source = self.__get_data_from_source(query.source_filters) if query.source_filters else []

        # If citations are provided, make query with its filters
        df_target = self.__get_data_from_citations(query.target_filters) if query.citation_filters else []    
            
        # Merge results. If source and citations provided at the same time, an AND operation is done so that
        # only common conditions are filtered
        df_merged = self.__merge_source_and_target(df_source, df_target) 
        
        # If no aggregations required, return current merged dataframe
        if not query.aggregations: 
            return df_merged
        
        # Apply aggregations over results
        df_aggregated = self.__query_aggregations(df_merged, query.aggregations)
        
        # Provide output
        output = query.output
        if output.distinct:
            return df_aggregated.columns(output.columns).unique() 
        
        return df_aggregated.columns(output.columns)      
        
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
    
    def __merge_source_and_target(self, df_source: pl.DataFrame, df_citation: pl.DataFrame) -> pl.DataFrame:
        if len(df_source) == 0:
            return df_citation
        
        if len(df_citation) == 0:
            return df_source
        
        return df_source.join(df_citation, on=["source_title", "source_year", "citation_title", "citation_year"])
    
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
                
        expr = self.__translate_vector_db_filters(filters) if filters.filters else ""
        df_source_papers = self.query_vector_db(
            text=filters.text,
            expr=expr,
            # top_k=100
        )
        
        conferences = df_source_papers.select("Conference").unique().to_dicts()
        commitees_per_conference = []
        for conference in conferences:
            relational_db = self.RELATIONAL_DB["CONFERENCE_MAPPING"][conference["Conference"]]
            client = self.relational_db_client(relational_db)
            client.connect()
            committees = client.query("""
                SELECT c_inst.year, comm.name, inst.name, inst.country
                FROM Committee_Institution AS c_inst 
                INNER JOIN Committee as comm
                ON comm.id_committee = c_inst.id_committee
                INNER JOIN Institutions as inst
                ON inst.id_institution = c_inst.id_institution
                """)
            client.close()
            
            for committee in committees:
                commitees_per_conference.append({
                    "Conference": conference["Conference"],
                    "Year": committee[0],
                    "Committee": {
                        "committee_name": committee[1],
                        "committee_institution": committee[2],
                        "committee_country": committee[3],
                    }
                })
                
        df_committees_per_conference = pl.DataFrame(commitees_per_conference)\
            .with_columns(
                pl.col("Year").cast(pl.String).alias("Year")
            )\
            .group_by("Year", "Conference")\
            .agg([pl.col("Committee")])
            
        df_source_papers = df_source_papers.join(
            df_committees_per_conference, 
            on=["Conference", "Year"],
            how="left"
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
    
    def __translate_relational_db_filters(self, ) -> str:
        pass
    
    
    # class CitationParameters(BaseModel):
    # group_by: Optional[list[str]] = []
    # aggregation: Optional[AggregationOperations] = None
    # descending: Optional[bool] = True    
    # limit: int = 10
    def __query_aggregations(self, df: pl.DataFrame, parameters: AggregationParameters) -> pl.DataFrame:
        
        if not parameters:
            return df
        
        aggregations = parameters.aggregations
        sorting = parameters.sort
        limit = parameters.limit
        
        # If aggregations provided, aggregate fields
        df_agg = df
        if aggregations:
            for aggregation in aggregations:
                expr = []
                if aggregation == AggregationOperations.count:
                    expr.append(pl.len().alias("count"))
                elif aggregation == AggregationOperations.mean:
                    expr.append(pl.mean().alias("mean")) 
            df_agg = df_agg.group_by(parameters.group_by).agg(expr)
        
        # If sorting provided, make sort
        df_sort = df_agg
        if parameters.sort:
            sort_columns = []
            descending = []
            for s in sorting:
                sort_columns.append(s.field)
                descending.append(s.descending)
                
            descending = [sort.descending for sort in sorting]
            
            df_sort = (
                df_sort.group_by(parameters.group_by)
                    .agg(expr)
                    .sort(sort_columns, descending=descending)
            )
        
        if limit:           
            return df_sort.limit(limit)
        
        return df_sort 