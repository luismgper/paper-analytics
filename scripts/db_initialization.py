from pathlib import Path
from src.papers.io.db import Milvus, Neo4j, DataType
from src.papers.domain.db_loader  import get_extended_crawler_data_df, load_data_to_vector_db, get_key_concepts
from dotenv import load_dotenv
import os

def start():
    print("DB initializing script started")

    load_dotenv()
    # TODO variables de entorno y no usar caminos relativos
    EXTENDED_CRAWLER_DATA_PATH = os.getenv("EXTENDED_CRAWLER_DATA_PATH")
    MILVUS_DB = os.getenv("MILVUS_DB")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")
    # NEO4J_URI = os.getenv("NEO4J_URI")
    # NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    # NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    # NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

    print(MILVUS_COLLECTION)

    extended_crawler_data_path = Path(EXTENDED_CRAWLER_DATA_PATH)
    extended_crawler_data_cols = [
        "Abstract",
        "Authors and Institutions",
        "Citations S2",
        # "DOI Number",
        # "OpenAlex Link",
        # "OpenAlex Referenced Works",
        "S2 Paper ID",
        "TLDR",
        "Title",
        "Conference",
        "Year"        
    ]
    collection_schema_field=[
        {"field_name": "id", "datatype": DataType.INT64, "is_primary": True},
        {"field_name": "Abstract", "datatype": DataType.VARCHAR, "max_length": 65535, "is_primary": False},
        {"field_name": "AbstractVector", "datatype": DataType.FLOAT_VECTOR,"is_primary": False},
        # {"field_name": "Authors_and_Institutions", "datatype": DataType.ARRAY, "is_primary": False},
        # {"field_name": "Citations_S2", "datatype": DataType.ARRAY, "is_primary": False},
        {"field_name": "S2_Paper_ID", "datatype": DataType.VARCHAR, "max_length": 100, "is_primary": False},
        {"field_name": "TLDR", "datatype": DataType.VARCHAR, "max_length": 65535, "is_primary": False},
        {"field_name": "TLDRVector", "datatype": DataType.FLOAT_VECTOR, "is_primary": False},
        {"field_name": "Title", "datatype": DataType.VARCHAR, "max_length": 65535, "is_primary": False},
        {"field_name": "TitleVector", "datatype": DataType.FLOAT_VECTOR, "is_primary": False},
        {"field_name": "Year", "datatype": DataType.VARCHAR, "max_length": 4, "is_primary": False},
        {"field_name": "KeyConcepts", "datatype": DataType.VARCHAR, "max_length": 65535, "is_primary": False},
        {"field_name": "KeyConceptsVector", "datatype": DataType.FLOAT_VECTOR, "is_primary": False},
        {"field_name": "Conference", "datatype": DataType.VARCHAR, "max_length": 100, "is_primary": False},
    ]

    print("Reading papers from crawler data...")

    # Get dataframe with needed data from crawler json
    df_abstracts = get_extended_crawler_data_df(extended_crawler_data_path, extended_crawler_data_cols)

    print("Extracting key concepts...")
    df_abstracts_with_keywords = get_key_concepts(df_abstracts)

    print("Initializing vector database...")

    # Initialize vector db client, db and collection
    milvus_client = Milvus(MILVUS_DB, MILVUS_COLLECTION, schema_fields=collection_schema_field, new_collection=True)    
    load_data_to_vector_db(df_data_to_insert=df_abstracts_with_keywords, milvus_client=milvus_client)

    # Initialize graph db client
    # neo4j_client = Neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)

