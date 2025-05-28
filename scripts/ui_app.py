import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from typing import List
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

def run_query(query: str):
    from src.papers.io import db
    from src.papers.domain.rag import RAG
    load_dotenv()
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")
    MILVUS_ALIAS = os.getenv("MILVUS_ALIAS")
    MILVUS_HOST = os.getenv("MILVUS_HOST")
    MILVUS_PORT = os.getenv("MILVUS_PORT")
    
    milvus_client = db.Milvus(
        collection=MILVUS_COLLECTION if MILVUS_COLLECTION else "", 
        alias=MILVUS_ALIAS if MILVUS_ALIAS else "",
        host=MILVUS_HOST if MILVUS_HOST else "",
        port=MILVUS_PORT if MILVUS_PORT else "", 
        new_collection=False
    )
    
    neo4j_client = db.Neo4j("bolt://localhost:7687", "neo4j", "password", "middleware")
    # rag_client = RAG(milvus_client=milvus_client, neo4j_client=neo4j_client, llm="gemma3:4b")    
    rag_client = RAG(milvus_client=milvus_client, neo4j_client=neo4j_client, llm="qwen3")    
        
    # response, analytics = rag_client.query(query)
    response = rag_client.query(query)

    report = f"Query: {query}\n\nResults:\n{response}"
    return report#, analytics


def show_plot(df: pl.DataFrame):
    df_pd = df.to_pandas()
    fig, ax = plt.subplots()
    ax.bar(df_pd["Category"], df_pd["Value"])
    ax.set_title("Polars Chart (via Matplotlib)")
    ax.set_ylabel("Value")
    st.pyplot(fig)


def main():
    st.set_page_config(layout="wide")
    st.title("Paper Research topic searcher")
    query = st.text_input("Enter your research query:", key="query_input")

    if st.button("Submit Query"):
        with st.spinner("Processing..."):
            # report, analytics = run_query(query)
            report = run_query(query)

        st.subheader("Report:")
        # st.text_area("Results", report, height=300)
        st.markdown(report)

        st.subheader("Cited papers")
        # st.dataframe(analytics["processed_papers"], use_container_width=False)
        
        # st.subheader("Cross conference citations")
        # st.dataframe(analytics["cross_conference_citations"])
        # st.dataframe(analytics["cross_conference_citations_test"]["contingency_table"])
        
        # st.subheader("Cross conference citations")
        # st.dataframe(analytics["cross_country_citations"])        
        # st.dataframe(analytics["cross_country_citations_test"]["contingency_table"])
        # st.markdown(analytics["cross_conference_citations_response"]["message"]["content"])
        # show_plot(df)

    st.markdown("---")
    st.caption("You can submit another query at any time.")


if __name__ == "__main__":
    main()