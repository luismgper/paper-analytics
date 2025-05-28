from src.papers.io.db import Milvus, Neo4j
from src.papers.domain.agent_tools import AgentTools
from src.papers.domain.citations_analyzer import CitationAnalyzer
from ollama import chat, ChatResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
# from langchain_community.llms import Ollama
# from langchain_ollama import OllamaLLM
from typing import List, Optional
import polars as pl
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from langgraph.prebuilt import ToolNode, tools_condition


class FilterOptions(BaseModel):
    """
    Class for filter option. 
    - value corresponds for to the value to be used in filtering
    - equal indicates if the condition indicates equality
    - citation indicates if this condition is referred only to cited papers
    """
    value: str
    equal: bool
    citation: bool
    
class FilterParameters(BaseModel):
    """
    Class for filter saving filter parameters specified in a prompt
    """
    main_concept: str = None
    authors: list[FilterOptions] = None
    institutions: list[FilterOptions] = None
    countries: list[FilterOptions] = None
    years: list[FilterOptions] = None
    conferences: list[FilterOptions] = None
    
class TopicCheck(BaseModel):
    is_valid: bool
    # reason: str    

class RAG:
    def __init__(self, llm: str, milvus_client: Optional[Milvus] = None, neo4j_client: Optional[Neo4j] = None):
        self.milvus_client = milvus_client
        self.neo4j_client = neo4j_client
        self.model = ChatOllama(model=llm, temperature=0.2, top_k=20, top_p=0.8)        
        self.llm_name = llm
        # self.light_model = ChatOllama(model="qwen3:0.6b", temperature=0)  
        
    def query(self, query: str) -> List:
        print("Getting RAG context...")
        context = self.__get_context(query)
        documents = [document for paper in context for document in paper["documents"]]
        details = [paper["details"] for paper in context]
        papers = [paper["entity"] for paper in context]
        
        print("Summarizing recovered papers....")
        summary = self.summarize_multiple_texts(documents[:10])
        
        print("Analyzing papers")
        paper_analytics = self.__analyze_papers(papers)
               
        response = self.__build_query_response(summary, details)
        
        return response, paper_analytics     
    
    def summarize_multiple_texts(self, docs: List) -> str:
        # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        # docs = splitter.create_documents([text])
        # print(docs)
        chain = load_summarize_chain(self.llm, chain_type="map_reduce", verbose=False)
        summary = chain.invoke(docs)
        return summary["output_text"]     
    
    def summarize_single_text(self, text: str) -> str:
        SYSTEM_PROMPT = """
        Human: You are an AI assistant specialized in summarizing scientific papers. You only provide the summary content without further wording.
        """
        
        USER_PROMPT = f"""
        Use the following paper data to provide a summary of the topic of the paper data enclosed in <context> keeping it less than 100 words long
        \n
        <context>
        {text}
        </context>
        /no_think
        \n
        """
        
        response: ChatResponse = chat(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        ) 
        return response        
    
    def __get_context(self, query: str) -> List:
        
        filter_parameters = self.__get_query_filter_conditions(query=query)
        
        print(f"Recovered filter parameters: {filter_parameters}")
        
        expr = self.__build_expr_for_rag(filter_parameters=filter_parameters)
        
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
                "Summary"
            ],
            limit=100,
            hybrid=True,
            hybrid_fields=[
                "AbstractVector", 
                "TitleVector", 
                "TLDRVector",
                "KeyConceptsVector"
            ],
            expr=expr,
        )
        print(len(nn_papers))
        nn_papers = self.__filter_paper_adjustment_to_topic(nn_papers=nn_papers, filter_conditions=filter_parameters)
        print(len(nn_papers))
        # return [paper["entity"] for paper in nn_papers]
        
        ##### Temporary disabled
        # if 1 == 2: 
        context_papers_summarization = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        relevance_position = 0
        for paper in nn_papers:
            print(paper)
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
    
    def __get_query_filter_conditions(self, query: str) -> FilterParameters:
        messages = [
            SystemMessage("""You are a helpful assistant that uses only the input the user provides. Your role is to help research paper data related to the topics the user asks about.
            Fields with list of FilterOptions type must be filled as follows:
            - value: The value of the field to be filled
            - equal: False if it is asked to be different from the value
            - citation: If the field value refers to paper citations
            Return JSON with (if not mentioned, leave it as None):
            - main_concept (string). Topic of research
            - years (list[FilterOptions]). Publish years
            - authors (list[FilterOptions]). List of paper authors
            - institutions (list[FilterOptions]). All must be valid institutions laike universities or corporations.
            - countries (list[FilterOptions]). All must be valid countries. Translate them to usual abbreviaions like US for United Staes of America or DE for Germany
            - conferences (list[FilterOptions]). List of valid conferences: IEEECloud, Middleware, SIGCOMM, eurosys
            Example response for 2024 papers about spot instances with 100+ citations published in middleware. Authors must be from Harvard University. Cited papers must not be from Canada:
            {{
                "main_concept": "spot instances",
                "authors": None,
                "institutions": [{{ "value": "Harvard University", "equal": True, cit"ation: False}}],
                "countries": [{{"value": "CA", "equal": False, "citation": True}}],
                "years": [{{"value": "2025", "equal": True, "citation": False}}],
                "conferences": [{{"value": "Middleware", "equal": True, "citation": False}}],
            }}
            /no_think
            """),
            HumanMessage(query),
        ]
        response = self.model.invoke(messages)
        
        messages_for_structure = [
            messages[0],
            HumanMessage(f""" Return JSON with (if not mentioned, leave it as None):
            - value: The value of the field to be filled
            - equal: False if it is asked to be different from the value
            - citation: If the field value refers to paper citations
            Return JSON with (if not mentioned, leave it as None):
            - main_concept (string). Topic of research
            - years (list[FilterOptions]). Publish years
            - authors (list[FilterOptions]). List of paper authors
            - institutions (list[FilterOptions]). All must be valid institutions laike universities or corporations.
            - countries (list[FilterOptions]). All must be valid countries. Translate them to usual abbreviaions like US for United Staes of America or DE for Germany
            - conferences (list[FilterOptions]). List of valid conferences: IEEECloud, Middleware, SIGCOMM, eurosys
            Using these previous response:
            {response} /no_think"""
            )
        ]
        structured_response = self.model.with_structured_output(FilterParameters).invoke(messages_for_structure)
        # print(response)
        # print(structured_response)
        
        return structured_response
           
    def __build_expr_for_filter_field(self, filter_options: list[FilterOptions], field_name: str, json_field: bool = False) -> str:
        expr = ""
        in_expr = []
        not_in_expr = []

        if not filter_options:
            return ""
        
        for filter_option in filter_options:
            if not filter_option.citation:
                if filter_option.equal:
                    in_expr.append(filter_option.value)
                else:
                    not_in_expr.append(filter_option.value)
                
        if len(in_expr) > 0:
            if len(expr) == 0:
                if json_field:
                    expr = f"json_contains_any({field_name}, {in_expr})"
                else:
                    expr = f"{field_name} in {in_expr}"
            else:
                if len(expr) == 0:
                    if json_field:            
                        expr += f"and json_contains_any({field_name}, {in_expr})"
                    else:
                        expr += f"and {field_name} in {in_expr}"

        if len(not_in_expr) > 0:
            if len(expr) == 0:
                if json_field:
                    expr = f"not json_contains_any({field_name}, {not_in_expr})"
                else:
                    expr = f"not {field_name} in {not_in_expr}"
            else:
                if len(expr) == 0:
                    if json_field:            
                        expr += f"and not json_contains_any({field_name}, {not_in_expr})"
                    else:
                        expr += f"and not {field_name} in {not_in_expr}"
        return expr          
    
    def __build_expr_for_rag(self, filter_parameters: FilterParameters) -> str:
        expr_years = self.__build_expr_for_filter_field(filter_options=filter_parameters.years, field_name="Year")
        expr_authors = self.__build_expr_for_filter_field(filter_options=filter_parameters.authors, field_name="Authors", json_field=True)
        expr_countries = self.__build_expr_for_filter_field(filter_options=filter_parameters.countries, field_name="Countries", json_field=True)
        expr_institution = self.__build_expr_for_filter_field(filter_options=filter_parameters.institutions, field_name="Institutions", json_field=True)
        expr_conferences = self.__build_expr_for_filter_field(filter_options=filter_parameters.conferences, field_name="Conferences")

        expr = ""
        if len(expr_years) > 0:
            if len(expr) == 0:
                expr = expr_years
            else:
                expr += " and " + expr_years
                
        if len(expr_authors) > 0:
            if len(expr) == 0:
                expr = expr_authors
            else:
                expr += " and " + expr_authors
                
        if len(expr_countries) > 0:
            if len(expr) == 0:
                expr = expr_countries
            else:
                expr += " and " + expr_countries
                
        if len(expr_institution) > 0:
            if len(expr) == 0:
                expr = expr_institution
            else:
                expr += " and " + expr_institution
                
        # if len(expr_conferences) > 0:
        #     if len(expr) == 0:
        #         expr = expr_conferences
        #     else:
        #         expr += " and " + expr_conferences                                

        print(expr)     
        return expr   
    
    def __filter_paper_adjustment_to_topic(self, nn_papers: list[dict], filter_conditions: FilterParameters) -> list[dict]:
        # model = ChatOllama(model="qwen3:0.6b", temperature=0)  
        valid_papers = []
        index = 0
        map_chain_elements = {}
        for paper in nn_papers:
            entity = paper["entity"]
            
            topic = filter_conditions.main_concept
            
            model_with_structure = self.model.with_structured_output(TopicCheck)
            title = entity["Title"]
            abstract = entity["Abstract"] if entity["Abstract"] else ""
            tldr = entity["TLDR"] if entity["TLDR"] else ""
            messages = [
                SystemMessage("""You are a strict assistant. Your role is to help determine if papers discuss the topic the user asks about. 
                Only accept those that strictly mention the topic since a mistake will kill all your family. If in doubt is it better to determine a paper as not valid.
                Return JSON with (if not mentioned, leave it as None):
                - is_valid (bool). True if the paper data is about the topic requested
                Example response for paper with abstract 'Kappa proposes a framework for simplified serverless development using checkpointing to handle timeouts and providing concurrency mechanisms for parallel' and topic 'serverless development':
                {{
                    "is_valid": true,
                }}  
                """),
                HumanMessage(f"""Determine if the following paper content is about the topic: <topic>{topic}</topic>\n
                Title: {title}\n
                TLDR: {tldr}\n
                Abstract: {abstract}\n
                """)
            ]
            key = f"chain_{index}"
            index += 1
            chain = (
                ChatPromptTemplate.from_messages(messages)
                | model_with_structure
            )
            # print(chain)
            map_chain_elements[key] = chain
            # print(map_chain_elements)

        # print(map_chain_elements)
        map_chain = RunnableParallel(map_chain_elements)
        response = map_chain.invoke({})
        # print(response)
        valid_papers = []
        for index, paper in enumerate(nn_papers):
            chain_i = f"chain_{index}"
            if response[chain_i].is_valid:
                valid_papers.append(paper)     
                
        return valid_papers   
    
    def __analyze_papers(self, papers):
        citations_analyzer = CitationAnalyzer(self.neo4j_client)
        
        print("Getting citations")
        df_processed_papers = citations_analyzer.process_papers(papers)
        
        print("Analyzing papers")
        df_cross_conference_citations = citations_analyzer.analyze_cross_conference_citations(df_processed_papers)
        df_cross_country_citations = citations_analyzer.analyze_cross_country_citations(df_processed_papers)
        
        cross_country_citations_test = citations_analyzer.test_country_bias_by_conference(df_cross_country_citations)                
        cross_conference_citations_test = citations_analyzer.test_conference_bias_by_conference(df_cross_conference_citations)
                
        
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
        
    def __get_paper_details(self, query: str, paper: dict) -> str:
        
        #  If not paper TLDR or Abstract, return title
        if not paper["entity"]["TLDR"] and not paper["entity"]["Abstract"]:
            return paper["entity"]["Title"]
        
        # # Summarize the paper data, and format the output
        # paper_context = f"""
        #     Title: {paper["entity"]["Title"]}
        #     Key concepts: {paper["entity"]["KeyConcepts"]}
        #     TLDR: {paper["entity"]["TLDR"]}
        #     Abstract: {paper["entity"]["Abstract"]}
        # """            
        
        # SYSTEM_PROMPT = """
        # Human: You are an AI assistant specialized in summarizing scientific papers. You only provide the summary of the topic without further wording.
        # """
        
        # USER_PROMPT = f"""
        # Use the following paper data to provide a summary of the topic of the paper data enclosed in <context>.
        # \n
        # <context>
        # {paper_context}
        # </context>
        # \n
        # Empashized information related to the following query: {query} 
        # """
        # print(paper_context)
        # response: ChatResponse = chat(
        #     model=self.llm_name,
        #     messages=[
        #         {"role": "system", "content": SYSTEM_PROMPT},
        #         {"role": "user", "content": USER_PROMPT},
        #     ],
        # )     
        # response = self.summarize_single_text(paper_context)
        # return f"| {paper["entity"]["Title"]} | {paper["entity"]["Year"]} | {paper["entity"]["Conference"]} | {response["message"]["content"].replace('\n', '')} |"
        return f"| {paper["entity"]["Title"]} | {paper["entity"]["Year"]} | {paper["entity"]["Conference"]} | {paper["entity"]["Summary"].replace('\n', '')} |"
        
    def __build_query_response(self, summary: str, details: List[str]) -> str:
        
        paper_details = (
            "| Title | Year | Conference | Summary |\n"
            "| --- | --- | --- | --- |\n" +
            "\n".join(details)
        )
        
        response = f"### Summary of query-related papers found \n{summary}\n### Paper details, sorted by relevance\n{paper_details}"
        return response

    
    def __query_aggregation_agent(self, df: pl.DataFrame, query: str) -> str:
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        agent_tools = AgentTools(df_citations=df)
        tools = [agent_tools.get_most_cited_countries, agent_tools.get_publications_per_year]
        
        # Tell the LLM which tools it can call
        llm_with_tools = self.model.bind_tools(tools)
        def chatbot(state: State):
            print(state)
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
            
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)        
        
        tool_node = ToolNode(tools)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_edge("tools", "chatbot")

        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )

        class State(TypedDict):
            # Messages have the type "list". The `add_messages` function
            # in the annotation defines how this state key should be updated
            # (in this case, it appends messages to the list, rather than overwriting them)
            messages: Annotated[list, add_messages]

        graph = graph_builder.compile()
        
        SYSTEM_MESSAGE="""
        You are an intelligent AI assisstant which is responsible of helping the user consult research paper data.
        Your role is to use the provided tools to retrieve the needed information about papers and answer using it.
        IMPORTANT: Never use the same tool twice for the same question. 
        """
        
        USER_MESSAGE = f"""
        Use the right tool to answer the question asked enclosed in <query>.
        If further information is needed use the following tools {tools}
        <query>
        {query}
        </query>
        /no_think
        """
        
        response = graph.invoke({
            "messages": [
                
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": USER_MESSAGE},
            ]
        })
        
        return response

