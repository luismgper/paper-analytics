import polars as pl
from typing import List, Dict
from src.papers.io.db import Milvus, Neo4j
from tqdm import tqdm
import scipy.stats as stats

class CitationAnalyzer:
    # Mapping of paper conference field to corresponding graph DB
    conf_db_map = {
        "ccgrid": "ccgrid",
        "cloud": "socc",
        "europar": "europar",
        "eurosys": "eurosys",
        "ic2e": "ic2e",
        "icdcs": "icdcs",
        "IEEEcloud": "ieeecloud",
        "middleware": "middleware",
        "nsdi": "nsdi",
        "sigcomm": "sigcomm",
    }    
    
    def __init__(self, graph_client: Neo4j):
        self.graph_client = graph_client

    def close(self):
        self.graph_client.close()

    def __map_conferences_to_databases(self, papers: List[Dict]) -> Dict[str, str]:
        return {paper["Title"]: self.conf_db_map[paper["Conference"]] for paper in papers}

    def __fetch_citation_data(self, title: str, year: int, database: str) -> List[Dict]:
        query = """
        MATCH (p:Paper {title: $title, year: $year})
        OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
        OPTIONAL MATCH (cited)-[:HAS_INSTITUTION]->(inst:Institution)-[:LOCATED_IN]->(country:Country)
        RETURN p.title as paper_title, p.year as paper_year, p.PredominantCountry as predominant_country, p.Authors as authors,
        p.conference as conference, 
        cited.title AS cited_title, cited.PredominantCountry as cited_predominant_country, cited.conference AS cited_conference, 
        cited.Authors as cited_authors, country.name AS cited_country, inst.name as cited_institution
        """
        result = self.graph_client.run_query(
            query=query,
            database=database,
            parameters={"title": title, "year": year}
        )
        return result

    def process_papers(self, papers: List[Dict]) -> pl.DataFrame:
        conf_map = self.__map_conferences_to_databases(papers)
        records = []

        for i, paper in enumerate(tqdm(papers, desc="Retrieving citations")):
            title = paper["Title"]
            year = paper["Year"]
            conference = paper["Conference"]
            authors = paper["Authors"]
            institutions = paper["Institutions"]
            summary = paper["Summary"]
            abstract = paper.get("Abstract", "")
            tldr = paper.get("TLDR", "")
            
            db = conf_map.get(title)
            if not db:
                continue

            citations = self.__fetch_citation_data(title, year, db)
            for citation in citations:
                if citation["cited_title"]:  # filter missing data
                    records.append({
                        "source_title": title,
                        "source_year": year,
                        "source_conference": conference,
                        "source_predominant_country": citation["predominant_country"],
                        "source_authors": authors,
                        "source_institutions": institutions,
                        "source_summary": summary,
                        "source_tldr": tldr,
                        "source_abstract": abstract,
                        "cited_title": citation["cited_title"],
                        "cited_predominant_country": citation["cited_predominant_country"],
                        "cited_conference": citation["cited_conference"] or "UNKNOWN",
                        "cited_country": citation["cited_country"] or "UNKNOWN"
                    })

        return pl.DataFrame(records)

    def analyze_cross_conference_citations(self, df: pl.DataFrame) -> pl.DataFrame:
        print("Analyzing cross conference citations...")
        return (
            df.select([
                "source_title", 
                "source_year",
                "source_conference",
                "cited_title",
                "cited_conference",
                "cited_predominant_country"
            ])
            .unique()
            .group_by(["source_conference", "cited_conference"])
            .agg([
                pl.count().alias("citation_count")
            ])
            .sort("citation_count", descending=True)
        )

    def analyze_cross_country_citations(self, df: pl.DataFrame) -> pl.DataFrame:
        print("Analyzing cross country citations...")
        return (
            df.filter(pl.col("cited_country") != "UNKNOWN")
            .select([
                "source_title", 
                "source_year", 
                "source_conference", 
                "source_predominant_country", 
                "cited_title", 
                "cited_conference", 
                pl.col("cited_country").list.explode()
            ])
            .unique()
            .group_by(["source_conference", "cited_country"])
            .agg([
                pl.count().alias("citation_count")
            ])
            .sort("citation_count", descending=True)
        )
        
    def test_country_bias_by_conference(self, df: pl.DataFrame) -> Dict[str, any]:
        print("Performing hypothesis test for country citation bias by conference...")
        MINIMUM_HITS_PER_CELL = 5
        
        df_other_countries = (
            df.filter(pl.col("citation_count") < MINIMUM_HITS_PER_CELL)
            .select("cited_country")
            .unique()
        )
        other_countries = [cited_country["cited_country"] for cited_country in df_other_countries.to_dicts()]

        contingency_df = (
            df.with_columns([
                pl.when(pl.col("cited_country").is_in(other_countries))
                .then(pl.lit("OTHER"))
                .otherwise(pl.col("cited_country"))
                .alias("cited_country")
            ])
            .pivot(
                values="citation_count", 
                index="source_conference", 
                columns="cited_country", 
                aggregate_function="first"
            )
            .with_columns(pl.all().fill_null(0))
        )        
    
        contingency_pd_df = contingency_df.to_pandas().set_index("source_conference")
        chi2, p, dof, expected = stats.chi2_contingency(contingency_pd_df)

        return {
            "test": "chi2_country_bias_by_conference",
            "chi2_statistic": chi2,
            "p_value": p,
            "degrees_of_freedom": dof,
            "expected_frequencies": expected,
            "contingency_table": contingency_pd_df
        }
        
    def test_conference_bias_by_conference(self, df: pl.DataFrame) -> Dict[str, any]:
        print("Performing hypothesis test for inter-conference citation bias...")
        
        MINIMUM_HITS_PER_CELL = 5
        
        df_other_conferences = (
            df.filter(pl.col("citation_count") < MINIMUM_HITS_PER_CELL)
            .select("cited_conference")
            .unique()
        )
        other_conferences = [cited_conference["cited_conference"] for cited_conference in df_other_conferences.to_dicts()]
                
        contingency_df = (
            df.with_columns([
                pl.when(pl.col("cited_conference").is_in(other_conferences))
                .then(pl.lit("OTHER"))
                .otherwise(pl.col("cited_conference"))
                .alias("cited_conference")
            ])
            .pivot(
                values="citation_count", 
                index="source_conference", 
                columns="cited_conference", 
                aggregate_function="first"
            )
            .fill_null(0)
        )

        contingency_pd = contingency_df.to_pandas().set_index("source_conference")
        chi2, p, dof, expected = stats.chi2_contingency(contingency_pd)
        
        return {
            "test": "chi2_conference_bias_by_conference",
            "chi2_statistic": chi2,
            "p_value": p,
            "degrees_of_freedom": dof,
            "expected_frequencies": expected,
            "contingency_table": contingency_pd
        }        
