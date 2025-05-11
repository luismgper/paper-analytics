from src.papers.io.db import Milvus, Neo4j
from ollama import chat, ChatResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from typing import List

class RAG:
    def __init__(self, milvus_client: Milvus, neo4j_client: Neo4j, llm: str):
        self.milvus_client = milvus_client
        self.neo4j_client = neo4j_client
        self.llm = OllamaLLM(model=llm)        
        
    def query(self, query: str) -> str:
        context = self.__get_context(query)
        summary = self.__summarize(context)
        output = {
            "context": context,
            "summary": summary
        }
        return output["summary"]
            
        # SYSTEM_PROMPT = """
        # Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
        # """

        # USER_PROMPT = f"""
        # Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        # <context>
        # {context}
        # </context>
        # <question>
        # {query}
        # </question>
        # """
        # response: ChatResponse = chat(
        #     model=self.llm,
        #     messages=[
        #         {"role": "system", "content": SYSTEM_PROMPT},
        #         {"role": "user", "content": USER_PROMPT},
        #     ],
        # )
        # return response["message"]["content"]        
    
    def __summarize(self, docs: List) -> str:
        # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        # docs = splitter.create_documents([text])
        # print(docs)
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        summary = chain.invoke(docs)
        return summary        
    
    def __get_context(self, query: str) -> List:
        
        # An hybrid search will be done. The top papers for each kind of embedding will be retrieved.
        # Then a reranking will be done to retrieve the most similar ones taking into account every 
        # kind of embedding
        nn_papers = self.milvus_client.search(
            text=query,
            output_fields=[
                "Title",
                "TLDR",
                "Abstract",
                "KeyConcepts"
            ],
            limit=10,
            hybrid=True,
            hybrid_fields=[
                "AbstractVector", 
                "TitleVector", 
                "TLDRVector",
                "KeyConceptsVector"
            ]
        )
        
        context_papers = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        relevance_position = 0
        for paper in nn_papers:
            relevance_position += 1
            paper_context = f"""
                Title: {paper["entity"]["Title"]}
                TLDR: {paper["entity"]["TLDR"]}
                Abstract: {paper["entity"]["Abstract"]}
            """                  
            chunks = splitter.split_text(paper_context)            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "paper_id": relevance_position,
                        "chunk_index": i
                    }
                )
                context_papers.append(doc)                
            
        return context_papers
            # paper_context = f"""
            # #     Title: {paper["entity"]["Title"]}
            # #     Key concepts: {paper["entity"]["KeyConcepts"]}
            # #     TLDR: {paper["entity"]["TLDR"]}
            # #     Abstract: {paper["entity"]["Abstract"]}
            # # """      
            # summarized_papers.append(paper_context)
            # #     <context>
            # #     {paper_context}
            # #     </context>
            # #     Provide an output in the form of:
            # #     Title: <Provided paper title>
            # #     Summary: <Elaborated summary>
            # # """

            # USER_PROMPT = f"""
            #     Use the following pieces of information enclosed in <context> to extract relevant paper information and summarize it in less than 100 words.
            #     <context>
            #     {paper_context}
            #     </context>
            # """            
                    
            # response: ChatResponse = chat(
            #     model=self.llm,
            #     messages=[
            #         {"role": "system", "content": SYSTEM_PROMPT},
            #         {"role": "user", "content": USER_PROMPT},
            #     ],
            # )
            # print(response["message"]["content"])
            # summarized_papers.append(response["message"]["content"])                         
        
        # context = "\n".join(
        #     [f"Paper:\n {paper}" for paper in summarized_papers]
        # )
        # return context
        
        

    
    
          