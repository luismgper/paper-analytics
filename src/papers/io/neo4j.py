from neo4j import GraphDatabase
from typing import Any, Dict, List, Optional

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
            with self.driver.session() as session:
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