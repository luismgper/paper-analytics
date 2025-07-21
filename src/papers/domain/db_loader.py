import polars as pl
from pathlib import Path
import re
from tqdm import tqdm
from typing import List
from keybert import KeyBERT 
from src.papers.io.db import Milvus, Neo4j
from src.papers.utils.models import Models
from src.papers.domain.rag import RAG

def get_extended_crawler_data_df(extended_crawler_data_path: Path, selected_cols: List) -> pl.DataFrame:
    """
    Reads the provided path, searching for JSONs with paper data. JSON are expexted to 

    Args:
        extended_crawler_data_path (Path): Path where JSON to be read are stored
        selected_cols (List): Columns to read from JSON property keys

    Returns:
        pl.DataFrame: Dataframe translated paper data
    """
    # Get JSON files from extended crawler data path
    json_files = list(extended_crawler_data_path.glob("*.json"))
    flattened_dfs = []
    
    # For each file generate a polars dataframe formatted
    for file in json_files:
        # Extract conference name from filename
        conference = re.sub(r".*/|_extended_data\.json", "", str(file))

        # Read the file
        df = pl.read_json(file, infer_schema_length=100000)
        for year in df.columns:
            if year == "Conference":
                continue

            # Each column is a list of dicts (papers)
            df_year = (
                pl.DataFrame({
                    "paper": df[year].explode()
                })
                .unnest("paper")
                .with_columns([
                    pl.lit(conference).alias("Conference"),
                    pl.lit(year).alias("Year")
                ])
            )

            df_year_reduced = df_year.select([col for col in selected_cols if col in df_year.columns])
            flattened_dfs.append(df_year_reduced)
            
    # Merge all dataframes in a single one and filter only tha papers including an abstract
    df_final = (
        pl.concat(flattened_dfs, how="vertical").filter(pl.col("Abstract").is_not_null() | pl.col("TLDR").is_not_null()).limit(1000)
        .with_columns(
            pl.lit(False).alias("IsCitation")
        )
    )
    
    # Filter out entries with null Abstracts and return them
    return df_final 

def get_paper_citations_df(citations_crawler_data_path: Path) -> pl.DataFrame:
    """
    Reads cititations from raw JSONs and stores required columns to polars dataframe

    Args:
        graph_client (Neo4j): _description_

    Returns:
        pl.DataFrame: _description_
    """
    
    # Get JSON files from extended crawler data path
    json_files = list(citations_crawler_data_path.glob("*.json"))
    flattened_dfs = []

    # For each file generate a polars dataframe formatted
    for file in json_files:
        # Extract conference name from filename 

        # Read the file
        df_json = pl.read_json(file, infer_schema_length=100000)
        for paper in df_json.columns:
            df_paper = (
                pl.DataFrame({
                    "paper": df_json[paper].explode()
                })
                .unnest("paper")
                .with_columns(
                    pl.col("Venue").cast(pl.String).alias("Conference"),
                    pl.col("Year").cast(pl.String),
                )
            )
            flattened_dfs.append(df_paper)

    # Concatenate all DataFrames
    df_final = (
        pl.concat(flattened_dfs, how="vertical")
        .with_columns(
            pl.lit("").alias("Abstract"),
            pl.col("Authors").alias("Authors and Institutions"),
            pl.lit("").alias("TLDR"),
            pl.lit(True).alias("IsCitation")
        )
        .select(['Abstract', 'Authors and Institutions', 'TLDR', 'Title', 'Conference', 'Year', 'IsCitation']).limit(1000)
    )
            
    return df_final
  
    
def load_data_to_vector_db(df_data_to_insert: pl.DataFrame, milvus_client: Milvus, rag: RAG):
    """
    Loads provided paper data into Vector database

    Args:
        df_data_to_insert (pl.DataFrame): Data to insert in polars dataframe
        milvus_client (Milvus): Vector database client to use for insertion
        rag (RAG): Class to use to summarize paper data

    Returns:
        _type_: _description_
    """
    papers = df_data_to_insert.to_dicts()
    
    def summarize(text: str) -> str:
        # print(text)
        response = rag.summarize_single_text(text)
        print(f"Summary: {response["message"]["content"]}")
        return response["message"]["content"]
    
    for i, row in enumerate(tqdm(papers, desc="Creating embeddings")):
        
        data = []
        title = row["Title"] if row["Title"] else None
        abstract = row["Abstract"] if row["Abstract"] else None
        key_concepts = row["KeyConcepts"] if row["KeyConcepts"] else None
        tldr = row["TLDR"] if row["TLDR"] else None
        authors = []
        institutions = []
        countries = []
        if row["Authors and Institutions"]:
            for author in row["Authors and Institutions"]:
                authors.append(author["Author"])
                if author["Institutions"]:
                    for institution in author["Institutions"]:
                        institutions.append(institution["Institution Name"])
                        countries.append(institution["Country"])
        
        empty_embedding = milvus_client.emb_text("")
        data.append({
            # "S2_Paper_ID": row["S2 Paper ID"], 
            "Title": title,
            "TitleVector": milvus_client.emb_text(row["Title"]) if row["Title"] else empty_embedding,
            "Abstract": abstract, 
            "AbstractVector": milvus_client.emb_text(row["Abstract"]) if row["Abstract"] else empty_embedding, 
            "TLDR": tldr,
            "TLDRVector": milvus_client.emb_text(row["TLDR"]) if row["TLDR"] else empty_embedding,
            "Year": row["Year"],
            "AuthorsAndInstitutions": row["Authors and Institutions"],
            "Authors": authors,
            "Institutions": institutions,
            "Countries": countries,
            "KeyConcepts": key_concepts,
            "KeyConceptsVector": milvus_client.emb_text(row["KeyConcepts"]) if row["KeyConcepts"] else empty_embedding,
            "Conference": row["Conference"],
            "Summary": summarize(f"Title: {title}. TLDR: {tldr}. Abstract: {abstract}. Key concepts: {key_concepts}") if tldr != "" and abstract != "" else title,     
            "IsCitation": row["IsCitation"]   
        })         
        milvus_client.insert(data=data)
    print("Finished inserting")        
        
def get_key_concepts(df_abstracts: pl.DataFrame) -> pl.DataFrame:
    """
    For given dataframe of papers, generates key concepts field

    Args:
        df_abstracts (pl.DataFrame): Papers dataframe

    Returns:
        pl.DataFrame: Original dataframe + KeyConcepts field with keywords of title, abstract and TLDR
    """
    models = Models()
    concept_model = models.get_concept_extraction_model()
    kw_model = KeyBERT(model=concept_model)
    
    def extract_keywords(text: str) -> str:
        keywords_scored = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
        
        enum_keywords = ""
        for keyword, score in keywords_scored:
            enum_keywords = keyword if enum_keywords == "" else enum_keywords + ", " + keyword 
            
        return enum_keywords
    
    # def summarize(text: str) -> str:
    #     # print(text)
    #     response = rag.summarize_single_text(text)
    #     print(f"Summary: {response["message"]["content"]}")
    #     return response["message"]["content"]
    df_kw = df_abstracts.with_columns(
            (
                pl.col("Title").fill_null("") + " \n " +
                pl.col("Abstract").fill_null("") + " \n " +
                pl.col("TLDR").fill_null("")
            ).str.strip_chars(", ").alias("Content")
    ).with_columns(
        pl.when(
            pl.col("Content").is_not_null() & (pl.col("Content").str.strip_chars() != "")
        )\
            .then(pl.col("Content").map_elements(extract_keywords, return_dtype=pl.String))\
            .otherwise(None)\
            .alias("KeyConcepts")
    ).select(['Abstract', 'Authors and Institutions', 'TLDR', 'Title', 'Conference', 'Year', 'IsCitation', 'KeyConcepts'])     
        # pl.when(pl.col("Abstract").is_not_null())\
        #     .then(pl.col("Abstract").map_elements(extract_keywords, return_dtype=pl.String))\
        #     .otherwise(None)\
        #     .alias("Abstract concepts"),
            
        # pl.when(pl.col("TLDR").is_not_null())\
        #     .then(pl.col("TLDR").map_elements(extract_keywords, return_dtype=pl.String))\
        #     .otherwise(None)\
        #     .alias("TLDR concepts"),
        
        # pl.when(pl.col("Title").is_not_null())\
        #     .then(pl.col("Title").map_elements(extract_keywords, return_dtype=pl.String))\
        #     .otherwise(None)\
        #     .alias("Title concepts"),           
    # ).with_columns(
    #     (
    #         # pl.col("Title concepts").fill_null("") + ", "+\
    #         pl.col("Abstract concepts").fill_null("") + ", "+\
    #         pl.col("TLDR concepts").fill_null("")
    #     ).str.strip_chars(", ").alias("KeyConcepts")
    # ).with_columns(
    #     pl.concat_str(
    #         [   
    #             "Title: "+pl.col("Title").fill_null(""),
    #             "TLDR: "+pl.col("TLDR").fill_null(""),                
    #             "Abstract: "+pl.col("Abstract").fill_null(""),
    #             "Key concepts: "+pl.col("KeyConcepts").fill_null(""),             
    #             # f"Title: {pl.col("Title").fill_null("")}",
    #             # f"TLDR: {pl.col("TLDR").fill_null("")}",                
    #             # f"Abstract: {pl.col("Abstract").fill_null("")}",
    #             # f"Key concepts: {pl.col("KeyConcepts").fill_null("")}",
    #         ],
    #         separator=". ",
    #     ).alias("text_to_summarize")         
    # ).with_columns(
    #     pl.col("text_to_summarize").map_elements(summarize, return_dtype=pl.String)\
    #         .alias("Summary")
    # )
    return df_kw

