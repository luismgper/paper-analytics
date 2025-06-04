from src.papers.io import db
from pydantic import BaseModel, Field
from enum import Enum

class Operator(str, Enum):
    eq = "eq"
    lt = "lt"
    gt = "gt"
    le = "le"
    ge = "ge"
    ne = "ne"
    
class FilterCondition(BaseModel):
    field: str
    operator: Operator
    values: list[str]    
    
class SourceFilters(BaseModel):
    topic: str
    filters: list[FilterCondition]
    
class CitationFilters(BaseModel):
    topic: str
    filters: list[FilterCondition]
    
class QueryParameters(BaseModel):
    source_filters: SourceFilters
    citation_fitlers: CitationFilters
    
class MultiModalPaperQuery():
    def __init__(self, sql_client: db.SQLite, vector_client: db.MilvusClient, graph_client: db.Neo4j):
        self.sql_client = sql_client
        self.vector_client = vector_client
        self.graph_client = graph_client
    
    def query(self, query: QueryParameters):
        source_data = self.__get_source_query(query.source_filters) if query.source_filters else []
        target_data = self.__get_target_query(filters=query.target_parameters) if query.target_filters else []    
        