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
            # self.milvus_embedding_model = SentenceTransformer("allenai/specter2_base", device=self.device)
            # self.milvus_embedding_model = SentenceTransformer("sentence-transformers/allenai-specter", device=self.device)
            # self.milvus_embedding_model = SentenceTransformer("Muennighoff/SGPT-125M-weightedmean-nli-bitfit")
        return self.milvus_embedding_model
    
    def get_concept_extraction_model(self):
        if not self.concept_extraction_model:
            # self.concept_extraction_model = SentenceTransformer("allenai/scibert_scivocab_uncased", device=self.device)
            # tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            self.concept_extraction_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')            
        return self.concept_extraction_model
    
    # def get_llm_model(self):
    #     if not self.llm_model:
    #         self.concept_extraction_model = SentenceTransformer("sentence-transformers/allenai-specter", device=self.device)
    #     return self.concept_extraction_model 