from pymilvus import MilvusClient
# from typing import Callable
from src.utils.models import Models

class Milvus:
    
    def __init__(self, db: str, collection: str, new_collection: bool=False):
        self.milvus_client = MilvusClient(db)            
        self.model = Models().get_milvus_embedding_model()               
        
        self.collection_name = collection                  
        if new_collection:
            self.drop_collection()
            self.__create_collection()
        
    def drop_collection(self):
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)

    def insert(self, data: list):
        self.milvus_client.insert(collection_name=self.collection_name, data=data)
        
    def search(self, text: str, output_fields: list, limit=10):
        search_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[
                self.emb_text(text)
            ],  
            limit=limit, 
            search_params={"metric_type": "INNER", "params": {}},  # Inner product distance
            output_fields=output_fields, #["Abstract"],  # Return the text field
        )
        return search_res[0]
    
    def emb_text(self, text:str):
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()     
    
    def __create_collection(self, dimension):
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=self.model.get_sentence_embedding_dimension,
            primary_field_name="id",
            id_type="int",    
            metric_type="INNER",
            consistency_level="Strong",  # Strong consistency level
            auto_id=True
        )
        