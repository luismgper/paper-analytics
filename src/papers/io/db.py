# import abc
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker, connections
# from typing import Callable
from src.papers.utils.models import Models
from neo4j import GraphDatabase
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

class Milvus:
    
    def __init__(
        self, 
        collection: str, 
        alias: str,
        host: str,
        port: str,
        db: str=None, 
        schema_fields: List=None, 
        new_collection: bool=False, 
        metric_type: str="IP"
    ):
        self.alias = alias
        self.host = host
        self.port = port
        connections.connect(
            alias=self.alias,
            host=self.host,
            port=self.port
        )        
        self.db = db if db else f"http://{self.host}:{self.port}"
        self.milvus_client = MilvusClient(self.db)
        
        model = Models()
        self.model = model.get_milvus_embedding_model()               
        self.metric_type = metric_type
        self.collection_name = collection    
        self.schema_fields = schema_fields               
        if new_collection:
            self.drop_collection()
            self.__create_collection()
        
    def drop_collection(self):
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)

    def insert(self, data: list):
        self.milvus_client.insert(collection_name=self.collection_name, data=data)
        
    def search(self, text: str, output_fields: list, limit: int=10, hybrid: bool=False, hybrid_fields: List=[]):
        if hybrid:
            requests = []
            for field in hybrid_fields:
                search_param = {
                    "data": [self.emb_text(text)],
                    "anns_field": field,
                    "param": {
                        "metric_type": "IP",
                        "params": {"nprobe": 10}
                    },
                    "limit": limit,
                }                   
                requests.append(AnnSearchRequest(**search_param))
                
            ranker = RRFRanker()
            response = self.milvus_client.hybrid_search(
                collection_name=self.collection_name,
                reqs=requests,
                ranker=ranker,
                limit=limit,
                output_fields=output_fields,
            )          
            print(response)
            return response[0]
        else:
            search_res = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[
                    self.emb_text(text)
                ],  
                limit=limit, 
                anns_field="Abstract",
                search_params={"metric_type": self.metric_type, "params": {}},  # Inner product distance
                output_fields=output_fields, #["Abstract"],  # Return the text field
            )
            return search_res[0]
    
    def emb_text(self, text:str):
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()     
    
    def __create_collection(self):
        if self.schema_fields:
            self.__create_schema_and_index()
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                # dimension=self.model.get_sentence_embedding_dimension(),
                # primary_field_name="id",
                # id_type="int",    
                # metric_type=self.metric_type,
                # consistency_level="Strong",  # Strong consistency level
                # auto_id=True,
                schema=self.schema,
                index_params=self.index_params,
            )
        else:
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                dimension=self.model.get_sentence_embedding_dimension(),
                primary_field_name="id",
                id_type="int",    
                metric_type=self.metric_type,
                consistency_level="Strong",  # Strong consistency level
                auto_id=True
            )
    
    def __create_schema_and_index(self):
        self.schema = self.milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        index_fields = []
        for field in self.schema_fields:      
            if field["datatype"] == DataType.FLOAT_VECTOR or field["datatype"] == DataType.FLOAT16_VECTOR:
                self.schema.add_field(
                    field_name=field["field_name"], 
                    datatype=field["datatype"], 
                    is_primary=field["is_primary"], 
                    dim=self.model.get_sentence_embedding_dimension(),
                )                  
                index_fields.append(field)
            else:
                self.schema.add_field(**field)
                    # field_name=field["field_name"], 
                    # datatype=field["datatype"], 
                    # is_primary=field["is_primary"], 
                # )     
                                
        self.index_params = self.milvus_client.prepare_index_params()
        for field in index_fields:
            self.index_params.add_index(
                field_name=field["field_name"],
                index_name=field["field_name"]+"_index",
                index_type="AUTOINDEX",
                metric_type=self.metric_type,
            )
        return self.schema, self.index_params
    
class Neo4j:
    def __init__(self, uri: str, username: str, password: str, database: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password), database=self.database)

    def open(self):
        if not self.driver or self.is_driver_closed():
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password), database=self.database)

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None
            
    def is_driver_closed(self) -> bool:
        try:
            with self.driver.session():
                return False  # It's open and working
        except Exception:
            return True  # Probably closed            
        
    def run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None, write: bool = False) -> List[Dict[str, Any]]:
        """
        Runs a Cypher query. Set write=True for write transactions.
        Returns a list of dicts with the result.
        """
        def _run(tx):
            result = tx.run(query, parameters or {})
            return [record.data() for record in result]

        self.driver.open()
            
        with self.driver.session() as session:
            if write:
                return session.execute_write(_run)
            else:
                return session.execute_read(_run)        
        
    # def read(self, query, parameters={}):
    #     driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
    #     with driver.session() as session:
    #         result = session.execute_read(self.run_query, parameters)

    #     driver.close()
    #     return result        
    
    
    # def write(self, query, params={}):
    #     driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
    #     with driver.session() as session:
    #         result = session.execute_write(query, params)
        
    #     driver.close()
    #     return result    
            
    # def run_query(self, tx, query, parameters={}):
    #     return tx.run(query, parameters).data()
    
class SQLite:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def execute(self, query: str, params: Tuple = ()) -> None:
        if not self.cursor:
            raise RuntimeError("Database not connected.")
        self.cursor.execute(query, params)
        self.conn.commit()