# import streamlit as st
# import polars as pl
# import matplotlib.pyplot as plt
# from dotenv import load_dotenv
# import os
# from typing import List
# import torch
# torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# def run_query(query: str):
#     from src.papers.io import db
#     from src.papers.domain.rag import RAG
#     load_dotenv()
#     MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")
#     MILVUS_ALIAS = os.getenv("MILVUS_ALIAS")
#     MILVUS_HOST = os.getenv("MILVUS_HOST")
#     MILVUS_PORT = os.getenv("MILVUS_PORT")
    
#     milvus_client = db.Milvus(
#         collection=MILVUS_COLLECTION if MILVUS_COLLECTION else "", 
#         alias=MILVUS_ALIAS if MILVUS_ALIAS else "",
#         host=MILVUS_HOST if MILVUS_HOST else "",
#         port=MILVUS_PORT if MILVUS_PORT else "", 
#         new_collection=False
#     )
    
#     neo4j_client = db.Neo4j("bolt://localhost:7687", "neo4j", "password", "middleware")
#     # rag_client = RAG(milvus_client=milvus_client, neo4j_client=neo4j_client, llm="gemma3:4b")    
#     rag_client = RAG(milvus_client=milvus_client, neo4j_client=neo4j_client, llm="qwen3:30b-a3b")    
#     # rag_client = RAG(milvus_client=milvus_client, neo4j_client=neo4j_client, llm="qwen3")    
        
#     # response, analytics = rag_client.query(query)
#     response = rag_client.query(query)

#     report = f"Query: {query}\n\nResults:\n{response}"
#     return report#, analytics


# def show_plot(df: pl.DataFrame):
#     df_pd = df.to_pandas()
#     fig, ax = plt.subplots()
#     ax.bar(df_pd["Category"], df_pd["Value"])
#     ax.set_title("Polars Chart (via Matplotlib)")
#     ax.set_ylabel("Value")
#     st.pyplot(fig)


# def main():
#     st.set_page_config(layout="wide")
#     st.title("Paper Research topic searcher")
#     query = st.text_input("Enter your research query:", key="query_input")

#     if st.button("Submit Query"):
#         with st.spinner("Processing..."):
#             # report, analytics = run_query(query)
#             report = run_query(query)

#         st.subheader("Report:")
#         # st.text_area("Results", report, height=300)
#         st.markdown(report)

#         # st.subheader("Cited papers")
#         # st.dataframe(analytics["processed_papers"], use_container_width=False)
        
#         # st.subheader("Cross conference citations")
#         # st.dataframe(analytics["cross_conference_citations"])
#         # st.dataframe(analytics["cross_conference_citations_test"]["contingency_table"])
        
#         # st.subheader("Cross conference citations")
#         # st.dataframe(analytics["cross_country_citations"])        
#         # st.dataframe(analytics["cross_country_citations_test"]["contingency_table"])
#         # st.markdown(analytics["cross_conference_citations_response"]["message"]["content"])
#         # show_plot(df)

#     st.markdown("---")
#     st.caption("You can submit another query at any time.")


# if __name__ == "__main__":
#     main()




import streamlit as st
import polars as pl
from src.papers.domain.multimodal_paper_query import MultiModalPaperQuery, OutputParameters, CitationFilters, SourceFilters, QueryParameters, LogicConnector, ComparisonOperator, AggregationOperations, FilterCondition, SortOptions, AggregationParameters
import src.papers.domain.multimodal_paper_query as mpq
from src.papers.io.db import Milvus, Neo4j, SQLite

import os
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# # Pydantic models (assuming these are already defined in your project)
# class LogicConnector(str, Enum):
#     none = ""
#     and_op = "and"
#     or_op = "or"

# class ComparisonOperator(str, Enum):
#     eq = "eq"
#     lt = "lt"
#     gt = "gt"
#     le = "le"
#     ge = "ge"
#     ne = "ne"
    
# class AggregationOperations(str, Enum):
#     count = "count"
#     mean = "mean"
    
# class SortOptions(BaseModel):
#     field: str
#     descending: bool = True    
    
# class OutputParameters(BaseModel):
#     fields: list[str]
#     distinct: Optional[bool] = False
        
# class FilterCondition(BaseModel):
#     level: int = 0
#     connector: LogicConnector = LogicConnector.none
#     field: str
#     operator: ComparisonOperator
#     values: list[str]    
    
# class SourceFilters(BaseModel):
#     text: Optional[str] = None
#     filters: Optional[list[FilterCondition]] = None
    
# class CitationFilters(BaseModel):
#     text: Optional[str] = None
#     filters: Optional[list[FilterCondition]] = None
    
# class AggregationParameters(BaseModel):
#     group_by: Optional[list[str]] = []
#     aggregations: Optional[list[AggregationOperations]] = None
#     sort: Optional[list[SortOptions]] = None    
#     limit: int = 10       
    
# class QueryParameters(BaseModel):
#     source_filters: Optional[SourceFilters] = None
#     citation_filters: Optional[CitationFilters] = None
#     aggregations: Optional[AggregationParameters] = None
#     output: Optional[OutputParameters] = None

def main():
    st.set_page_config(layout="wide")
    st.title("Query Interface")
    st.markdown("Configure your query parameters below:")
    
    # Initialize session state for dynamic filter management
    if 'source_filters' not in st.session_state:
        st.session_state.source_filters = []
    if 'citation_filters' not in st.session_state:
        st.session_state.citation_filters = []
    
    # Source Filters Section
    st.header("Source Filters")
    use_source = st.checkbox("Enable Source Filters")
    
    source_text = None
    source_filter_conditions = None
    
    if use_source:
        source_text = st.text_input("Source Text Filter", placeholder="Enter text to search in source...")
        
        st.subheader("Source Filter Conditions")
        
        # Add filter button
        if st.button("Add Source Filter", key="add_source"):
            st.session_state.source_filters.append({})
        
        # Display existing filters
        source_filter_conditions = []
        for i, filter_data in enumerate(st.session_state.source_filters):
            with st.expander(f"Source Filter {i+1}", expanded=True):
                # col1, col2, col3 = st.columns([2, 2, 1])
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                # with col1:
                #     level = st.number_input("Level", min_value=0, value=0, key=f"source_level_{i}")
                #     connector = st.selectbox("Connector", ["", "and", "or"], key=f"source_connector_{i}")
                #     field = st.text_input("Field", key=f"source_field_{i}")
                
                # with col2:
                #     operator = st.selectbox("Operator", ["eq", "lt", "gt", "le", "ge", "ne"], key=f"source_operator_{i}")
                #     values_input = st.text_input("Values (comma-separated)", key=f"source_values_{i}")
                #     values = [v.strip() for v in values_input.split(",") if v.strip()]
                
                # with col3:
                #     if st.button("Remove", key=f"remove_source_{i}"):
                #         st.session_state.source_filters.pop(i)
                #         st.rerun()
                with col1:
                    level = st.number_input("Level", min_value=0, value=0, key=f"source_level_{i}")
                with col2:
                    connector = st.selectbox("Connector", ["", "and", "or"], key=f"source_connector_{i}")
                with col3:
                    field = st.text_input("Field", key=f"source_field_{i}")
                with col4:
                    operator = st.selectbox("Operator", ["eq", "lt", "gt", "le", "ge", "ne"], key=f"source_operator_{i}")
                with col5:
                    values_input = st.text_input("Values (comma-separated)", key=f"source_values_{i}")
                    values = [v.strip() for v in values_input.split(",") if v.strip()]
                with col6:
                    if st.button("Remove", key=f"remove_source_{i}"):
                        st.session_state.source_filters.pop(i)
                        st.rerun()                
                
                if field and operator and values:
                    source_filter_conditions.append(FilterCondition(
                        level=level,
                        connector=LogicConnector(connector),
                        field=field,
                        operator=ComparisonOperator(operator),
                        values=values
                    ))
    
    # Citation Filters Section
    st.header("Citation Filters")
    use_citation = st.checkbox("Enable Citation Filters")
    
    citation_text = None
    citation_filter_conditions = None
    
    if use_citation:
        citation_text = st.text_input("Citation Text Filter", placeholder="Enter text to search in citations...")
        
        st.subheader("Citation Filter Conditions")
        
        # Add filter button
        if st.button("Add Citation Filter", key="add_citation"):
            st.session_state.citation_filters.append({})
        
        # Display existing filters
        citation_filter_conditions = []
        for i, filter_data in enumerate(st.session_state.citation_filters):
            with st.expander(f"Citation Filter {i+1}", expanded=True):
                # col1, col2, col3 = st.columns([2, 2, 1])
                
                # with col1:
                #     level = st.number_input("Level", min_value=0, value=0, key=f"citation_level_{i}")
                #     connector = st.selectbox("Connector", ["", "and", "or"], key=f"citation_connector_{i}")
                #     field = st.text_input("Field", key=f"citation_field_{i}")
                
                # with col2:
                #     operator = st.selectbox("Operator", ["eq", "lt", "gt", "le", "ge", "ne"], key=f"citation_operator_{i}")
                #     values_input = st.text_input("Values (comma-separated)", key=f"citation_values_{i}")
                #     values = [v.strip() for v in values_input.split(",") if v.strip()]
                
                # with col3:
                #     if st.button("Remove", key=f"remove_citation_{i}"):
                #         st.session_state.citation_filters.pop(i)
                #         st.rerun()
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    level = st.number_input("Level", min_value=0, value=0, key=f"source_level_{i}")
                with col2:
                    connector = st.selectbox("Connector", ["", "and", "or"], key=f"source_connector_{i}")
                with col3:
                    field = st.text_input("Field", key=f"source_field_{i}")
                with col4:
                    operator = st.selectbox("Operator", ["eq", "lt", "gt", "le", "ge", "ne"], key=f"source_operator_{i}")
                with col5:
                    values_input = st.text_input("Values (comma-separated)", key=f"source_values_{i}")
                    values = [v.strip() for v in values_input.split(",") if v.strip()]
                with col6:
                    if st.button("Remove", key=f"remove_source_{i}"):
                        st.session_state.source_filters.pop(i)
                        st.rerun()                   
                
                if field and operator and values:
                    citation_filter_conditions.append(FilterCondition(
                        level=level,
                        connector=LogicConnector(connector),
                        field=field,
                        operator=ComparisonOperator(operator),
                        values=values
                    ))
    
    # Aggregations Section
    st.header("Aggregations")
    use_aggregations = st.checkbox("Enable Aggregations")
    
    aggregations = None
    if use_aggregations:
        col1, col2 = st.columns(2)
        
        with col1:
            group_by_input = st.text_input("Group By Fields (comma-separated)", placeholder="field1, field2, ...")
            group_by = [f.strip() for f in group_by_input.split(",") if f.strip()]
            
            agg_operations = st.multiselect("Aggregation Operations", ["count", "mean"])
            
        with col2:
            limit = st.number_input("Limit", min_value=1, value=10)
            
            # Sort options
            st.subheader("Sort Options")
            sort_field = st.text_input("Sort Field")
            sort_descending = st.checkbox("Descending", value=True)
        
        sort_options = [SortOptions(field=sort_field, descending=sort_descending)] if sort_field else None
        
        aggregations = AggregationParameters(
            group_by=group_by,
            aggregations=[AggregationOperations(op) for op in agg_operations] if agg_operations else None,
            sort=sort_options,
            limit=limit
        )
    
    # Output Parameters Section
    st.header("Output Parameters")
    use_output = st.checkbox("Configure Output", value=True)
    
    output = None
    if use_output:
        col1, col2 = st.columns(2)
        
        with col1:
            output_fields_input = st.text_input("Output Fields (comma-separated)", placeholder="field1, field2, ...")
            output_fields = [f.strip() for f in output_fields_input.split(",") if f.strip()]
        
        with col2:
            distinct = st.checkbox("Distinct Results")
        
        if output_fields:
            output = OutputParameters(fields=output_fields, distinct=distinct)
    
    # Build Query Parameters
    st.header("Execute Query")
    
    if st.button("Build Query Parameters", type="primary"):
        # Build source filters
        source_filters_obj = None
        if use_source and (source_text or source_filter_conditions):
            source_filters_obj = SourceFilters(
                text=source_text if source_text else None,
                filters=source_filter_conditions if source_filter_conditions else None
            )
        
        # Build citation filters
        citation_filters_obj = None
        if use_citation and (citation_text or citation_filter_conditions):
            citation_filters_obj = CitationFilters(
                text=citation_text if citation_text else None,
                filters=citation_filter_conditions if citation_filter_conditions else None
            )
        
        # Build query parameters
        query_params = QueryParameters(
            source_filters=source_filters_obj,
            citation_filters=citation_filters_obj,
            aggregations=aggregations,
            output=output
        )
        
        # Display the built query parameters
        st.subheader("Generated Query Parameters")
        st.json(query_params.model_dump(exclude_none=True))
        
        # Store in session state for use
        st.session_state.query_params = query_params
        
        st.success("Query parameters built successfully!")
    
    # Execute Query Section (placeholder)
    if hasattr(st.session_state, 'query_params'):
        st.subheader("Execute Query")
        st.info("Query parameters are ready. You can now call your query method with these parameters.")
        
        # Example of how to use the parameters
        # st.code("""
        # # Example usage:
        # query_params = st.session_state.query_params
        # result_df = your_query_object.query(query_params)
        # st.dataframe(result_df)
        # """)
        
        # # If you have access to your query object, you could execute it here:
        if st.button("Execute Query"):
            print("query call")
            # try:
            MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")
            MILVUS_ALIAS = os.getenv("MILVUS_ALIAS")
            MILVUS_HOST = os.getenv("MILVUS_HOST")
            MILVUS_PORT = os.getenv("MILVUS_PORT")
            NEO4J_URI = os.getenv("NEO4J_URI")
            NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
            NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
            NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

            milvus_client = Milvus(
                    collection=MILVUS_COLLECTION, 
                    alias=MILVUS_ALIAS,
                    host=MILVUS_HOST,
                    port=MILVUS_PORT,
                )   
            neo4j_client = Neo4j(
                uri=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE
            )

            query_client = MultiModalPaperQuery(relational_db_client=SQLite, vector_db_client=milvus_client, graph_db_client=neo4j_client)

            result_df = query_client.query(st.session_state.query_params)
            st.dataframe(result_df)
            # except Exception as e:
            #     st.error(f"Error executing query: {str(e)}")

if __name__ == "__main__":
    main()