from pathlib import Path
from src.papers.io.db import Milvus, Neo4j, DataType
from src.papers.domain.db_loader  import get_extended_crawler_data_df, get_paper_citations_df, load_data_to_vector_db, get_key_concepts
from src.papers.domain.rag import RAG
from dotenv import load_dotenv
import polars as pl
import os

def start():
    print("DB initializing script started")

    load_dotenv()
    # TODO variables de entorno y no usar caminos relativos
    EXTENDED_CRAWLER_DATA_PATH = os.getenv("EXTENDED_CRAWLER_DATA_PATH")
    CITATIONS_CRAWLER_DATA_PATH = os.getenv("CITATIONS_CRAWLER_DATA_PATH")
    # MILVUS_DB = os.getenv("MILVUS_DB")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")
    MILVUS_ALIAS = os.getenv("MILVUS_ALIAS")
    MILVUS_HOST = os.getenv("MILVUS_HOST")
    MILVUS_PORT = os.getenv("MILVUS_PORT")
    # NEO4J_URI = os.getenv("NEO4J_URI")
    # NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    # NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    # NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

    print(MILVUS_COLLECTION)

    extended_crawler_data_path = Path(EXTENDED_CRAWLER_DATA_PATH)
    citations_crawler_data_path = Path(CITATIONS_CRAWLER_DATA_PATH)
    extended_crawler_data_cols = [
        "Abstract",
        "Authors and Institutions",
        # "Citations S2",
        # "DOI Number",
        # "OpenAlex Link",
        # "OpenAlex Referenced Works",
        # "S2 Paper ID",
        "TLDR",
        "Title",
        "Conference",
        "Year"        
    ]
    collection_schema_field=[
        {"field_name": "id", "datatype": DataType.INT64, "is_primary": True},
        {"field_name": "Abstract", "datatype": DataType.VARCHAR, "max_length": 65535, "is_primary": False, "nullable": True},
        {"field_name": "AbstractVector", "datatype": DataType.FLOAT_VECTOR,"is_primary": False},
        {"field_name": "AuthorsAndInstitutions", "datatype": DataType.JSON, "is_primary": False, "nullable": True},
        {"field_name": "Authors", "datatype": DataType.JSON, "is_primary": False, "nullable": True},
        {"field_name": "Institutions", "datatype": DataType.JSON, "is_primary": False, "nullable": True},
        {"field_name": "Countries", "datatype": DataType.JSON, "is_primary": False, "nullable": True},
        # {"field_name": "Citations_S2", "datatype": DataType.ARRAY, "is_primary": False},
        # {"field_name": "S2_Paper_ID", "datatype": DataType.VARCHAR, "max_length": 100, "is_primary": False, "nullable": True},
        {"field_name": "TLDR", "datatype": DataType.VARCHAR, "max_length": 65535, "is_primary": False, "nullable": True},
        {"field_name": "TLDRVector", "datatype": DataType.FLOAT_VECTOR, "is_primary": False},
        {"field_name": "Title", "datatype": DataType.VARCHAR, "max_length": 65535, "is_primary": False, "nullable": True},
        {"field_name": "TitleVector", "datatype": DataType.FLOAT_VECTOR, "is_primary": False},
        {"field_name": "Year", "datatype": DataType.VARCHAR, "max_length": 4, "is_primary": False, "nullable": True},
        {"field_name": "KeyConcepts", "datatype": DataType.VARCHAR, "max_length": 65535, "is_primary": False, "nullable": True},
        {"field_name": "KeyConceptsVector", "datatype": DataType.FLOAT_VECTOR, "is_primary": False, "nullable": True},
        {"field_name": "Conference", "datatype": DataType.VARCHAR, "max_length": 100, "is_primary": False},
        {"field_name": "Summary", "datatype": DataType.VARCHAR, "max_length": 65535, "is_primary": False},
        {"field_name": "IsCitation", "datatype": DataType.BOOL, "is_primary": False},
    ]

    print("Reading papers from crawler data...")

    # Get dataframe with needed data from crawler json
    df_abstracts = get_extended_crawler_data_df(extended_crawler_data_path, extended_crawler_data_cols)
    df_citations = get_paper_citations_df(citations_crawler_data_path)

    print("Extracting key concepts...")

    df_abstracts_with_keywords = get_key_concepts(df_abstracts)
    df_to_insert = pl.concat([df_abstracts_with_keywords, df_citations.with_columns(pl.lit("").alias("KeyConcepts"))], how="vertical")
    print(df_to_insert.head(100))
    
    print("Initializing vector database...")

    # Initialize vector db client, db and collection
    milvus_client = Milvus(
        # db=MILVUS_DB, 
        collection=MILVUS_COLLECTION, 
        alias=MILVUS_ALIAS,
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        schema_fields=collection_schema_field, 
        new_collection=True
    )    
    # milvus_client = Milvus(MILVUS_DB, MILVUS_COLLECTION, schema_fields=collection_schema_field, new_collection=True)    
    rag = RAG(llm="llama3.2")
    load_data_to_vector_db(df_data_to_insert=df_to_insert, milvus_client=milvus_client, rag=rag)

    # Initialize graph db client
    # neo4j_client = Neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)

if __name__ == "__main__":
    start()