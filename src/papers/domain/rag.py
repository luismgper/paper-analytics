from src.papers.io.db import Milvus, Neo4j
from ollama import chat, ChatResponse

class RAG:
    def __init__(self, milvus_client: Milvus, neo4j_client: Neo4j, llm: str):
        self.milvus_client = milvus_client
        self.neo4j_client = neo4j_client
        self.llm = llm        
        
    def query(self, query: str) -> str:
        context = self.__get_context(query)
        
        SYSTEM_PROMPT = """
        Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
        """

        USER_PROMPT = f"""
        Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        <context>
        {context}
        </context>
        <question>
        {query}
        </question>
        """
        response: ChatResponse = chat(
            model=self.llm,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        )
        return response["message"]["content"]        
        
    
    def __get_context(self, query: str) -> str:
        
        # An hybrid search will be done. The top papers for each kind of embedding will be retrieved.
        # Then a reranking will be done to retrieve the most similar ones taking into account every 
        # kind of embedding
        nn_papers_title = self.milvus_client.search(
            text=query,
            output_fields=[
                "Title",
                "TLDR",
                "Abstract",
            ],
            filter="vectorvectorField == 'Title'",
            limit=10
        )
        nn_papers_tldr = self.milvus_client.search(
            text=query,
            output_fields=[
                "Title",
                "TLDR",
                "Abstract",
            ],
            filter="vectorvectorField == 'TLDR'",
            limit=10
        )
        nn_papers_abstract = self.milvus_client.search(
            text=query,
            output_fields=[
                "Title",
                "TLDR",
                "Abstract",
            ],
            filter="vectorvectorField == 'Abstract'",
            limit=10
        )
        print(nn_papers_title)
        print(nn_papers_tldr)
        print(nn_papers_abstract)
        
                                                
        
        # context = "\n".join(
        #     [paper["entity"]["Abstract"] for paper in nn_papers]
        # )
        # return context
        
        

    
    
          