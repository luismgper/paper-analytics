import polars as pl
from pathlib import Path
import re
from tqdm import tqdm
from typing import List
from keybert import KeyBERT 
from src.papers.io.db import Milvus
from src.papers.utils.models import Models
from src.papers.domain.rag import RAG

def get_extended_crawler_data_df(extended_crawler_data_path: Path, selected_cols: List) -> pl.DataFrame:
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
    df_final = pl.concat(flattened_dfs, how="vertical").filter(pl.col("Abstract").is_not_null() | pl.col("TLDR").is_not_null())#.limit(100)
    # Filter out entries with null Abstracts and return them
    return df_final 
    
def load_data_to_vector_db(df_data_to_insert: pl.DataFrame, milvus_client: Milvus, rag: RAG):
    papers = df_data_to_insert.to_dicts()
    
    def summarize(text: str) -> str:
        # print(text)
        response = rag.summarize_single_text(text)
        print(f"Summary: {response["message"]["content"]}")
        return response["message"]["content"]
    
    data = []
    for i, row in enumerate(tqdm(papers, desc="Creating embeddings")):
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
            
        data.append({
            "S2_Paper_ID": row["S2 Paper ID"], 
            "Title": title,
            "TitleVector": milvus_client.emb_text(row["Title"]) if row["Title"] else milvus_client.emb_text(""),
            "Abstract": abstract, 
            "AbstractVector": milvus_client.emb_text(row["Abstract"]) if row["Abstract"] else milvus_client.emb_text(""), 
            "TLDR": tldr,
            "TLDRVector": milvus_client.emb_text(row["TLDR"]) if row["TLDR"] else milvus_client.emb_text(""),
            "Year": row["Year"],
            "AuthorsAndInstitutions": row["Authors and Institutions"],
            "Authors": authors,
            "Institutions": institutions,
            "Countries": countries,
            "KeyConcepts": key_concepts,
            "KeyConceptsVector": milvus_client.emb_text(row["KeyConcepts"]) if row["KeyConcepts"] else milvus_client.emb_text(""),
            "Conference": row["Conference"],
            "Summary": summarize(f"Title: {title}. TLDR: {tldr}. Abstract: {abstract}. Key concepts: {key_concepts}")        
        })         
        milvus_client.insert(data=data)
        data = []
        # if len(data) == 10:
        #     print("Inserting vector data...")
        #     milvus_client.insert(data=data)
        #     data = []
            
            
        # if row["Abstract"]:
        #     data.append({
        #         "S2 Paper ID": row["S2 Paper ID"], 
        #         "vector": milvus_client.emb_text(row["Abstract"]), 
        #         "vectorField": "Abstract",
        #         "Title": row["Title"] if row["Title"] else None,
        #         "Abstract": row["Abstract"] if row["Abstract"] else None, 
        #         "TLDR": row["TLDR"] if row["TLDR"] else None,
        #         "Year": row["Year"],
        #         "Authors anmodeld Institutions": row["Authors and Institutions"],
        #         "Conference": row["Conference"]        
        #     }) 
            
        # if row["Title"]:
        #     data.append({
        #         "S2 Paper ID": row["S2 Paper ID"], 
        #         "vector": milvus_client.emb_text(row["Title"]), 
        #         "vectorField": "Title",
        #         "Title": row["Title"] if row["Title"] else None,
        #         "Abstract": row["Abstract"] if row["Abstract"] else None, 
        #         "TLDR": row["TLDR"] if row["TLDR"] else None,
        #         "Year": row["Year"],
        #         "Authors and Institutions": row["Authors and Institutions"],
        #         "Conference": row["Conference"]        
        #     })             
        
        # if row["TLDR"]:
        #     data.append({
        #         "S2 Paper ID": row["S2 Paper ID"], 
        #         "vector": milvus_client.emb_text(row["TLDR"]), 
        #         "vectorField": "TLDR",
        #         "Title": row["Title"] if row["Title"] else None,
        #         "Abstract": row["Abstract"] if row["Abstract"] else None, 
        #         "TLDR": row["TLDR"] if row["TLDR"] else None,
        #         "Year": row["Year"],
        #         "Authors and Institutions": row["Authors and Institutions"],
        #         "Conference": row["Conference"]        
        #     }) 
    print("Finished inserting")        
        
def get_key_concepts(df_abstracts: pl.DataFrame):
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
            # pl.col("Title concepts").fill_null("") + ", "+\
            pl.col("Title").fill_null("") + " \n  "+\
            pl.col("Abstract").fill_null("") + " \n  "+\
            pl.col("TLDR").fill_null("")
        ).str.strip_chars(", ").alias("Content")
    ).with_columns(
        pl.when(pl.col("Content").is_not_null())\
            .then(pl.col("Content").map_elements(extract_keywords, return_dtype=pl.String))\
            .otherwise(None)\
            .alias("KeyConcepts")
    )     
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