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
        self.llm_name = llm
        
    def query(self, query: str) -> str:
        context = self.__get_context(query)
        documents = [document for paper in context for document in paper["documents"]]
        details = [paper["details"] for paper in context]
        
        summary = self.__summarize(documents)
        # output = {
        #     "context": context,
        #     "summary": summary
        # }
        
        response = self.__build_query_response(summary, details)
        return response
            
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
        return summary["output_text"]     
    
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
                "KeyConcepts",
                "Year",
                "Conference",
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
        
        context_papers_summarization = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        relevance_position = 0
        for paper in nn_papers:
            
            details = self.__get_paper_details(query, paper)
            
            relevance_position += 1
            paper_context = f"""
                Title: {paper["entity"]["Title"]}
                TLDR: {paper["entity"]["TLDR"]}
                Abstract: {paper["entity"]["Abstract"]}
            """                  
            
            chunks = splitter.split_text(paper_context)
            documents = []            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "paper_id": relevance_position,
                        "chunk_index": i
                    }
                )
                documents.append(doc)                
            
            context_papers_summarization.append({
                "documents": documents,
                "details": details,
            })
            
        return context_papers_summarization
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
    def __get_paper_details(self, query: str, paper: dict) -> str:
        
        # Summarize the paper data, and format the output
        paper_context = f"""
            Title: {paper["entity"]["Title"]}
            Key concepts: {paper["entity"]["KeyConcepts"]}
            TLDR: {paper["entity"]["TLDR"]}
            Abstract: {paper["entity"]["Abstract"]}
        """            
        
        SYSTEM_PROMPT = """
        Human: You are an AI assistant specialized in summarizing scientific papers. You only provide the summary of the topic without further wording.
        """
        
        USER_PROMPT = f"""
        Use the following paper data to provide a summary of the topic of the paper data enclosed in <context>.
        \n
        <context>
        {paper_context}
        </context>
        \n
        Empashized information related to the following query: {query} 
        """
        response: ChatResponse = chat(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        )
        print(response)     
        return f"""
        - Paper: {paper["entity"]["Title"]}
        - Year: {paper["entity"]["Year"]}
        - Conference {paper["entity"]["Conference"]}
        - Summary: {response["message"]["content"]}
        """
        
    def __build_query_response(self, summary: str, details: List[str]) -> str:
        
        paper_details = "\n".join(
            [details for details in details]
        )
            
        
        response = f"""
        ### Summary of query-related papers found
        {summary}
        
        ### Paper details, sorted by relevance
        {paper_details}
        """
        return response

    
    
          