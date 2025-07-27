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
    none = ""
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
    conference="Conference"
    
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
    
class Conference(str, Enum):
    """
    Possible conferences
    """     
    ccgrid="ccgrid" 
    cloud="cloud"
    europar="europar"
    eurosy="eurosys"
    ic2e="ic2e"
    icdcs="icdcs"
    IEEEcloud="IEEEcloud"
    middleware="middleware"
    nsdi="nsdi"
    sigcomm="sigcomm"    
    
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
    text: Optional[str] = Field(default=None, description="Topic regarding paper content of the query being made")
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
    year: Optional[str] = Field(default=None, description="Paper pubblishing year")
    
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
  
    GRAPH_DB_MAP_CONFERENCE_TO_DC = {
        Conference.ccgrid: "ccgrid",
        Conference.cloud: "socc",
        Conference.europar: "europar",
        Conference.eurosy: "eurosys",
        Conference.ic2e: "ic2e",
        Conference.icdcs: "icdcs",
        Conference.IEEEcloud: "ieeecloud",
        Conference.middleware: "middleware",
        Conference.nsdi: "nsdi",
        Conference.sigcomm: "sigcomm",            
    }
    
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
            "KeyConceptsVector",
        ],
        "LIST_FIELDS": [
            "Authors",
            "Institutions",
            "Countries",
        ],
        "LIST_FIELDS_EXPR_MAPPING": {
            ComparisonOperator.eq: "json_contains_any",
            ComparisonOperator.ne: "not json_contains_any",  
        },
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
        # "CONFERENCE_MAPPING": {
        #     "ccgrid": "../data/RelationalDB/ccgrid.db",
        #     "cloud": "../data/RelationalDB/cloud.db",
        #     "europar": "../data/RelationalDB/europar.db",
        #     "eurosys": "../data/RelationalDB/eurosys.db",
        #     "ic2e": "../data/RelationalDB/ic2e.db",
        #     "icdcs": "../data/RelationalDB/icdcs.db",
        #     "IEEEcloud": "../data/RelationalDB/IEEEcloud.db",
        #     "middleware": "../data/RelationalDB/middleware.db",
        #     "nsdi": "../data/RelationalDB/nsdi.db",
        #     "sigcomm": "../data/RelationalDB/sigcomm.db",
        # }        
    }
            
    def __init__(self, relational_db_client: db.SQLite, vector_db_client: db.Milvus, graph_db_client: db.Neo4j):
        """
        Constructor method

        Args:
            relational_db_client (db.SQLite): Relational DB client to use
            vector_db_client (db.Milvus): Vector DB client to use
            graph_db_client (db.Neo4j): Graph DB client to use
        """
        self.relational_db_client = relational_db_client
        self.vector_db_client = vector_db_client
        self.graph_db_client = graph_db_client
    
    def query(self, query: QueryParameters) -> pl.DataFrame:
        """
        General query method for papers and their citations

        Args:
            query (QueryParameters): Paper query parameters

        Returns:
            pl.DataFrame: Results of the query in polars dataframe format
        """
        
        # If source is provided, start by recovering source paper data.
        df_source = self.__get_data_from_source(query.source_filters) if query.source_filters else []

        # If citations are provided, make query with its filters
        df_target = self.__get_data_from_citations(query.citation_filters) if query.citation_filters else []    
            
        # Merge results. If source and citations provided at the same time, an AND operation is done so that
        # only common conditions are filtered
        apply_and = True if query.source_filters and query.citation_filters else False
        df_merged = self.__merge_source_and_target(df_source, df_target, apply_and) 
        
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
        
    def query_source(self, filters: SourceFilters) -> pl.DataFrame:
        if not filters:
            return []
        
        return self.__get_data_from_source(filters)
    
    def query_citation(self, filters: CitationFilters) -> pl.DataFrame:
        if not filters:
            return []
        
        return self.__get_data_from_citations(filters)    
        
    def query_vector_db(self, text: str, expr: str = "", top_k: int = 10, search_fields: list = VECTOR_DB["SEARCH_FIELDS"]) -> pl.DataFrame:
        """Retrieve data from vector database

        Args:
            text (str): text for similarity search
            expr (str, optional): expression to filter metadata. Defaults to "".
            top_k (int, optional): number of entries to retrieve. Defaults to 10.
            search_fields(list, optional): fields to use in search

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
        
    def get_conference_committees(self, conferences: Optional[list[Conference]]) -> pl.DataFrame:
        """
        Retrieve conference committees data from relational database
        
        Args:
            conferences (list[Conference]], optional): Conferences to retrieve

        Returns:
            pl.DataFrame: DataFrame with query results
        """
        

        conference_provided = (conferences and len(conferences) > 0 )
        query_conferences = (
            [conference for conference in conferences] 
                if conference_provided 
                else [conference.value for conference in Conference]
        )
        commitees_per_conference = []
        
        for conference in query_conferences:
            relational_db = self.RELATIONAL_DB["CONFERENCE_MAPPING"][conference]
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
                    "conference": conference,
                    "year": committee[0],
                    "committee": {
                        "committee_name": committee[1],
                        "committee_institution": committee[2],
                        "committee_country": committee[3],
                    }
                })        
        return pl.DataFrame(
            commitees_per_conference,
            schema={
                "conference": pl.String,
                "year": pl.String,
                "committee": pl.Struct([
                    pl.Field("committee_name", pl.String),
                    pl.Field("committee_institution", pl.String),
                    pl.Field("committee_country", pl.String)
                ])
            }
        )
        
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
        RETURN 
        p.title as source_title, 
        p.year as source_year, 
        p.PredominantCountry as source_country, 
        p.PredominantContinent as source_predominant_continent, 
        p.Authors as source_authors,
        p.conference as source_conference, 
        cited.title AS cited_title, 
        cited.year as cited_year,
        cited.PredominantCountry as cited_predominant_country, 
        cited.PredominantContinent as cited_predominant_continent, 
        cited.conference AS cited_conference, 
        cited.Authors as cited_authors, 
        country.name AS cited_country, 
        inst.name as cited_institution
        """
        df_results = self.query_graph_db(query=QUERY, parameters=parameters.model_dump())
        return df_results
    
    def get_citations_batch(self, papers_per_conference: dict) -> pl.DataFrame:
        
        QUERY = """
        UNWIND $papers AS paper_info
        MATCH (p:Paper {title: paper_info.title, year: paper_info.year})
        OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
        OPTIONAL MATCH (cited)-[:HAS_INSTITUTION]->(inst:Institution)-[:LOCATED_IN]->(country:Country)
        RETURN
            p.title as source_title,
            p.year as source_year,
            p.PredominantCountry as source_country,
            p.PredominantContinent as source_predominant_continent,
            p.Authors as source_authors,
            p.conference as source_conference,
            cited.title AS cited_title,
            cited.year as cited_year,
            cited.PredominantCountry as cited_predominant_country,
            cited.PredominantContinent as cited_predominant_continent,
            cited.conference AS cited_conference,
            cited.Authors as cited_authors,
            country.name AS cited_country,
            inst.name as cited_institution
        """
        
        results = []
        for conference in papers_per_conference:
            db = self.GRAPH_DB_MAP_CONFERENCE_TO_DC[conference["Conference"]]
            df_results = self.query_graph_db(
                query=QUERY, 
                parameters={"papers": conference["papers"]},
                databases=[db]
            )
            if len(df_results) > 0: 
                string_conversions = []
                for col in df_results.columns:
                    col_dtype = df_results[col].dtype
                    if not isinstance(col_dtype, pl.List):
                        # string_conversions.append(
                        #     pl.col(col).map_elements(lambda x: str(x) if x is not None else None, return_dtype=pl.String).alias(col)
                        # )
                    # else:
                        # Regular cast to string
                        string_conversions.append(pl.col(col).cast(pl.String))
                
                df_results = df_results.with_columns(string_conversions)
                results.append(df_results)            
        
        return pl.concat(results, how="vertical")        
    
    def get_citers(self, parameters: CitationParameters) -> pl.DataFrame:
        """Retrieve papers citing a paper
        Args:
            parameters (CitationParameters): Needed parameters for query related to the paper. Containing title and year

        Returns:
            pl.DataFrame: DataFrame with query results
        """
        
        filter_condition = "{title: $title, year: $year}" if parameters.year else "{title: $title}"
        
        # Cypher query to search cited papers from source paper
        QUERY = f"""MATCH (cited:Paper {filter_condition})
        OPTIONAL MATCH (p:Paper)-[:CITES]->(cited)
        OPTIONAL MATCH (cited)-[:HAS_INSTITUTION]->(inst:Institution)-[:LOCATED_IN]->(country:Country)
        OPTIONAL MATCH (p)-[:HAS_INSTITUTION]->(p_inst:Institution)-[:LOCATED_IN]->(p_country:Country)
        RETURN 
        p.title as source_title, 
        p.year as source_year, 
        p.PredominantCountry as source_predominant_country, 
        p.PredominantContinent as source_predominant_continent, 
        p.Authors as source_authors,
        p.conference as source_conference, 
        p_inst.name as source_institution, 
        p_country.name as source_country,
        cited.title AS cited_title, 
        cited.year as cited_year,
        cited.PredominantCountry as cited_predominant_country,
        cited.PredominantContinent as cited_predominant_continent,
        cited.conference AS cited_conference, 
        cited.Authors as cited_authors, 
        country.name AS cited_country, 
        inst.name as cited_institution
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
    
    def translate_vector_db_filters(self, filter: SourceFilters) -> str:
        """
        Translate given parameters into vector db string filter condition

        Args:
            filter (SourceFilters): FIlters to apply

        Returns:
            str: FIltering condition in string format
        """
        
        expr = ""
        for filter in filter.filters:
            # Translate filter condition to string
            expr_condition = self.__translate_vector_db_filter_condition(filter)
            
            # Concatenation of conditions depending on connector
            if filter.connector == LogicConnector.none:
                expr = expr_condition
            else:
                expr += f" {filter.connector.value} {expr_condition}"
                
            # If a parenthesis opening or closing is specified, add the corresponding parenthesis
            if filter.parenthesis == ParenthesisIndicator.open:
                expr = "( "+expr
            elif filter.parenthesis == ParenthesisIndicator.close:
                expr += " )"
                
        return expr      
    
    def __merge_source_and_target(self, df_source: pl.DataFrame, df_citation: pl.DataFrame, apply_join: Optional[bool] = False) -> pl.DataFrame:
        """
        Joins source and citations filters into a single dataframe. 
        If only one of them is provided, it is returning as the result.
        If both are indicated, they are joined returning only papers appearing in both dataframes 

        Args:
            df_source (pl.DataFrame): Source papers dataframe
            df_citation (pl.DataFrame): Cited papers dataframe
            apply_and (bool, optional): Indicator to force apply of joining, even if a dataframe is empty 

        Returns:
            pl.DataFrame: _description_
        """
        
        if len(df_source) == 0 and not apply_join:
            return df_citation
        
        if len(df_citation) == 0 and not apply_join:
            return df_source
        
        df_merged = df_source.with_columns(
            pl.col("source_title").cast(pl.String),
            pl.col("source_year").cast(pl.String),
            pl.col("cited_title").cast(pl.String),
            pl.col("cited_year").cast(pl.String),
        ).join(
            df_citation.with_columns(
                    pl.col("source_title").cast(pl.String),
                    pl.col("source_year").cast(pl.String),
                    pl.col("cited_title").cast(pl.String),
                    pl.col("cited_year").cast(pl.String),
            ),
            on=["source_title", "source_year", "cited_title", "cited_year"]
        )
        return df_merged
    
    def __get_citations_data_from_source(self, df_source: pl.DataFrame) -> pl.DataFrame:
        
        # UDF to use
        # def get_citations_call(x): 
        #     # print(x)
        #     return self.get_citations(
        #         CitationParameters(
        #             **{"title": x["Title"], "year": x["Year"]}
        #         )
        #     ).to_dicts()
        
        papers_per_conference = (
            df_source
            .with_columns(
                pl.col("Title").alias("title"),
                pl.col("Year").alias("year")
            )
            .select("title", "year", "Conference")
            .unique()
            .group_by("Conference")
            .agg([
                pl.struct(["title", "year"]).alias("papers")
            ])
            .to_dicts()
        )
        df_citations = self.get_citations_batch(papers_per_conference)
        
        df_papers_with_citations = (
            df_source
            .with_columns(
                pl.col("Conference").alias("source_conference"),
                pl.col("Title").alias("source_title"),
                pl.col("Year").alias("source_year"),
            )
            .join(
                other=df_citations,
                on=["source_title", "source_year", "source_conference"],
                how="left",
            )
            .select([
                'Committee',
                'source_title',
                'source_year',
                'source_country',
                'source_predominant_continent',
                'source_authors',
                'source_conference',
                'KeyConcepts',
                'Summary',
                'Authors',
                'Institutions',
                'Abstract',
                'TLDR',                
                'cited_title',
                'cited_year',
                'cited_predominant_country',
                'cited_conference',
                'cited_authors',
                'cited_country',
                'cited_institution'
            ])            
        )
        
        # df_papers_with_citations = (
        #     df_source.select(
        #         pl.all(),
        #         pl.struct("Title", "Year").alias("citers"),
        #     )
        #     .with_columns(
        #         pl.col("citers").map_elements(get_citations_call).alias("citations")
        #     )
        #     .explode("citations")
        #     .unnest("citations")
        #     .select([
        #         'Committee',
        #         'source_title',
        #         'source_year',
        #         'source_country',
        #         'source_predominant_continent',
        #         'source_authors',
        #         'source_conference',
        #         'KeyConcepts',
        #         'Summary',
        #         'Authors',
        #         'Institutions',
        #         'Abstract',
        #         'TLDR',                
        #         'cited_title',
        #         'cited_year',
        #         'cited_predominant_country',
        #         'cited_conference',
        #         'cited_authors',
        #         'cited_country',
        #         'cited_institution'
        #     ])
        # )
        # print("df_citation")
        # for paper in df_papers_with_citations.to_dicts()[:3]:
        #     print(paper)
        return df_papers_with_citations 
    
    def __get_citing_data_from_target(self, df_target: pl.DataFrame) -> pl.DataFrame:
        """
        Obtains papers citing the recovered cited papers

        Args:
            df_target (pl.DataFrame): DAtaframe of cited papers

        Returns:
            pl.DataFrame: Citations and their source citers
        """
       
        def get_citations_call(x): 
            """
            Retrieves citaters for paper

            Args:
                x (_type_): paper

            Returns:
                _type_: citer papers
            """
            parameters = {"title": x["Title"]}
            if x["Year"]:
                parameters["year"] = x["Year"]
                
            return self.get_citers(CitationParameters(**parameters)).to_dicts()
                
        df_papers_with_citations = (
            df_target.select(
                pl.all(),
                pl.struct("Title", "Year").alias("citations"),
            )
            .with_columns(
                pl.col("citations").map_elements(get_citations_call).alias("source")
            )
            .explode('source')
            .unnest('source')
            .select([
                'source_title',
                'source_year',
                'source_predominant_country',
                'source_predominant_continent',
                'source_authors',
                'source_conference',
                'source_institution',
                'source_country',
                'cited_title',
                'cited_year',
                'cited_predominant_country',
                'cited_predominant_continent',
                'cited_conference',
                'cited_authors',
                'cited_country',
                'cited_institution'                      
            ])
        )
        return df_papers_with_citations         
    
    def __get_data_from_source(self, filters: SourceFilters) -> pl.DataFrame:
        """
        Method to retrieve data form source papers

        Args:
            filters (SourceFilters): Filterin criteria to apply

        Returns:
            pl.DataFrame: DataFrame with query results
        """
        
        
        expr = ("IsCitation == False AND (" + self.translate_vector_db_filters(filters) + ")") if filters.filters else "IsCitation == False"
        print(expr)
        df_source_papers = self.query_vector_db(
            text=filters.text,
            expr=expr,
            top_k=100
        )
        print(f"source papers found in vector db: {len(df_source_papers)}")
        # for paper in df_source_papers.to_dicts()[:3]:
        #     print(paper)
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
        print(f"source papers with committees: {len(df_source_papers)}")
        # print("df_source_papers with committees")
        # for paper in df_source_papers.to_dicts()[:3]:
        #     print(paper)
            
        
        if len(df_source_papers) > 0:
            
            # Source data was recovered, then get its citation data
            df_source_with_citations = self.__get_citations_data_from_source(df_source_papers)      
            print(f"source papers with citations: {len(df_source_with_citations.select("source_title", "source_year", "source_conference").unique())}")
            return df_source_with_citations
        
        return []
    
    def __get_data_from_citations(self, filters: CitationFilters) -> pl.DataFrame:
        """
        Gets cited papers data from provided filters

        Args:
            filters (CitationFilters): Filter conditions for cited papers

        Returns:
            pl.DataFrame: Polars dataframe with results
        """
        expr = ("IsCitation == True AND (" + self.translate_vector_db_filters(filters) + ")") if filters.filters else "IsCitation == True"
        df_papers = self.query_vector_db(
            text=filters.text,
            expr=expr,
            search_fields=["TitleVector"]
            # top_k=100
        )

        if len(df_papers) > 0:
            
            # Papers found, cet citing papers
            df_target_with_citers = self.__get_citing_data_from_target(df_papers)
            return df_target_with_citers
        
        return []
    
    def __translate_vector_db_filter_condition( self, filter: FilterCondition) -> str:
        """
        Generates vector database string condition for a single filter condition over a metadata field

        Args:
            filter (FilterCondition): FIeld filtering condition

        Raises:
            TypeError: Exception when no valid values are provided for the condition

        Returns:
            str: String condition for fitlring vector databse
        """
        
        if filter.field in self.VECTOR_DB["LIST_FIELDS"]:
            op = self.VECTOR_DB["LIST_FIELDS_EXPR_MAPPING"][filter.operator]
            values_str = f"[{", ".join(f'"{value}"' for value in filter.values)}]"
            return f"{op}({filter.field.value}, {values_str})"
        else:
            if filter.operator in [ComparisonOperator.ge, ComparisonOperator.le, ComparisonOperator.gt, ComparisonOperator.lt] and len(filter.values) != 1:
                raise TypeError(f"Only a single item is allowed as values for inequality operator in field {filter.field}")
            
            op = self.VECTOR_DB["EXPR_MAPPING"][filter.operator]
            return f"{filter.field.value} {op} {filter.values}"

    def __query_aggregations(self, df: pl.DataFrame, parameters: AggregationParameters) -> pl.DataFrame:
        """
        Applies aggregations over provided data

        Args:
            df (pl.DataFrame): Paper data
            parameters (AggregationParameters): Provided aggregaiton parameters

        Returns:
            pl.DataFrame: Resulting aggregated data
        """
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