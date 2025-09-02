# from keybert import KeyBERT 
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

class Models:
    
    milvus_embedding_model = None
    concept_extraction_model = None
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_milvus_embedding_model(self):
        if not self.milvus_embedding_model:
            self.milvus_embedding_model = SentenceTransformer("intfloat/e5-base-v2", device=self.device)
        return self.milvus_embedding_model
    
    def get_concept_extraction_model(self):
        if not self.concept_extraction_model:
            self.concept_extraction_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')            
        return self.concept_extraction_model
