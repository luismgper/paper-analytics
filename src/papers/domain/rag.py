from src.papers.io.db import Milvus, Neo4j
from ollama import chat, ChatResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from typing import List
from src.papers.domain.citations_analyzer import CitationAnalyzer
import polars as pl

class RAG:
    def __init__(self, milvus_client: Milvus, neo4j_client: Neo4j, llm: str):
        self.milvus_client = milvus_client
        self.neo4j_client = neo4j_client
        self.llm = OllamaLLM(model=llm)        
        self.llm_name = llm
        
    def query(self, query: str) -> List:
        print("Getting RAG context...")
        context = self.__get_context(query)
        documents = [document for paper in context for document in paper["documents"]]
        details = [paper["details"] for paper in context]
        papers = [paper["entity"] for paper in context]
        
        print("Summarizing recovered papers....")
        summary = self.__summarize(documents)
        
        print()
        paper_analytics = self.__analyze_papers(papers)
        
        # output = {
        #     "context": context,
        #     "summary": summary
        # }
        
        response = self.__build_query_response(summary, details)
        
        return response, paper_analytics
            
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
        chain = load_summarize_chain(self.llm, chain_type="map_reduce", verbose=False)
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
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
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
                "entity": paper["entity"],
                "documents": documents,
                "details": details,
            })
            
        return context_papers_summarization
    
    def __analyze_papers(self, papers):
        citations_analyzer = CitationAnalyzer(self.neo4j_client)
        
        print("Getting citations")
        df_processed_papers = citations_analyzer.process_papers(papers)
        
        print("Analyzing papers")
        df_cross_conference_citations = citations_analyzer.analyze_cross_conference_citations(df_processed_papers)
        df_cross_country_citations = citations_analyzer.analyze_cross_country_citations(df_processed_papers)
        
        cross_country_citations_test = citations_analyzer.test_country_bias_by_conference(df_cross_country_citations)                
        cross_conference_citations_test = citations_analyzer.test_conference_bias_by_conference(df_cross_conference_citations)
        
        # SYSTEM_PROMPT = """
        # Human: You are an AI assistant specialized in reporting statistical findings without further wording or quesitons asked. Report as if the question was made by someone lacking statistics knowledge.
        # """        
        
        # USER_PROMPT = f"""
        # Use the following chi squared hypothesis test results to provide a summary of the findings done when checking citation patterns between conferences with the data enclosed in <context>. Provide interpretation over the data. Format the response in markdown.
        # \n
        # <context>
        # {cross_conference_citations_test}
        # </context>
        # \n
        # """
        # response: ChatResponse = chat(
        #     model=self.llm_name,
        #     messages=[
        #         {"role": "system", "content": SYSTEM_PROMPT},
        #         {"role": "user", "content": USER_PROMPT},
        #     ],
        # )
                
        
        analytics = {
            "processed_papers": df_processed_papers,
            "cross_conference_citations": df_cross_conference_citations,
            "cross_country_citations": df_cross_country_citations,
            "cross_country_citations_test": cross_country_citations_test,
            "cross_conference_citations_test": cross_conference_citations_test,
            # "cross_conference_citations_response": response,
        }
        print("Finished analyzing")
        return analytics
        
            # paper_context = f"""df_cross_country"]["KeyConcepts"]}
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
        
        #  If not paper TLDR or Abstract, return title
        if not paper["entity"]["TLDR"] and not paper["entity"]["Abstract"]:
            return paper["entity"]["Title"]
        
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
        print(paper_context)
        response: ChatResponse = chat(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        )     
        return f"| {paper["entity"]["Title"]} | {paper["entity"]["Year"]} | {paper["entity"]["Conference"]} | {response["message"]["content"].replace('\n', '')} |"
        
    def __build_query_response(self, summary: str, details: List[str]) -> str:
        
        paper_details = (
            "| Title | Year | Conference | Summary |\n"
            "| --- | --- | --- | --- |\n" +
            "\n".join(details)
        )
        
        response = f"### Summary of query-related papers found \n{summary}\n### Paper details, sorted by relevance\n{paper_details}"
        return response

    
    

