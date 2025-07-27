import src.papers.domain.multimodal_paper_query as mpq
from src.papers.io.db import Milvus, Neo4j, SQLite
import polars as pl
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import pycountry_convert as pc



class Continent(str, Enum):
    """
    Continent codes and their description
    """    
    AS = 'Asia',
    EU = 'Europe',
    NA = 'North America',
    SA = 'South America',
    AF = 'Africa',
    OC = 'Oceania',
    AN = 'Antarctica'

class PaperAnalytics():
    
    def __init__(self, query_client: mpq.MultiModalPaperQuery):
        self.query_client = query_client
        
    def query_paper_count_per_conference_continent_and_year(
        self, 
        conferences: Optional[list[mpq.Conference]] = None,
        years: Optional[list[str]] = None,
        continents: Optional[list[str]] = None
    ) -> pl.DataFrame:  
        """
        Method to query papers published per conference and continent
        Args:
            conferences (Optional[list[Conference]], optional): Conferences to filter by. Defaults to None.
            years (Optional[list[str]], optional): Continents to filter by. Defaults to None.
            continents (Optional[list[str]], optional): Years to filter by. Defaults to None.
            
        Returns:
            pl.DataFrame: Dataframe with calculated number of papers published in each conference,
                        year and continent
        """
        filter_conditions = []
        if conferences and len(conferences) > 0:
            filter_conditions.append(
                mpq.FilterCondition(
                    connector=mpq.LogicConnector.none.value,
                    field="Conference",
                    operator=mpq.ComparisonOperator.eq.value,
                    values=conferences
                )
            )
        
        if years and len(years) > 0:
            filter_conditions.append(
                mpq.FilterCondition(
                    connector=mpq.LogicConnector.and_op.value if len(filter_conditions) > 0 else mpq.LogicConnector.none.value,
                    field="Year",
                    operator=mpq.ComparisonOperator.eq.value,
                    values=years
                )
            )
                
        query_parameters = mpq.QueryParameters(
            source_filters=mpq.SourceFilters(
                text=None,
                filters=filter_conditions if len(filter_conditions) > 0 else None
            )
        )
        
        df_query = self.query_client.query(query_parameters)        
        df_result = (
            df_query
            .select([
                "source_title",
                "source_year",
                "source_predominant_continent",
                "source_conference"
            ])
            .explode("source_predominant_continent")        
            .filter(
                pl.when(continents is not None)
                .then(pl.col("source_predominant_continent").is_in(continents))
                .otherwise(True)            
            )
            .unique()
            .group_by("source_year", "source_conference", "source_predominant_continent")
            .agg([
                pl.len().alias("paper_count")
            ])
            # .sort("source_conference","source_year", "paper_count", descending=True)
        )
        
        # Ensure that there is no year that exists for a conference and continent combination but 
        # not for other. If it does not exist, it is filled with 0
        df_result_complete = (
            df_result
            .select("source_conference").unique()
            .join(df_result.select("source_year").unique(), how="cross")
            .join(df_result.select("source_predominant_continent").unique(), how="cross")
            .join(df_result, on=["source_conference", "source_year", "source_predominant_continent"], how="left")
            .with_columns(
                pl.col("paper_count").fill_null(0)
            )
            .sort("source_conference", "source_year", "paper_count", descending=True)
        )           
        
        return df_result_complete

    def query_paper_count_per_conference_and_continent(
        self,
        conferences: Optional[list[mpq.Conference]] = None,
        continents: Optional[list[str]] = None,    
    ) -> pl.DataFrame:
        """
        Get paper count per conference and continent

        Args:
            conferences (Optional[list[mpq.Conference]], optional): Conferences to filter by. Defaults to Field(default=None, description="Conference to use as filter").
            continents (Optional[list[str]], optional): Continent to filter by. Defaults to Field(default=None, description="Continents to filter by").

        Returns:
            pl.DataFrame: Results of query aggregated
        """
        
        df_result = (
            self.query_paper_count_per_conference_continent_and_year(conferences=conferences, continents=continents)
            .group_by("source_conference", "source_predominant_continent")
            .agg([
                pl.sum("paper_count")
            ])
            .sort("source_conference", "paper_count", descending=True)
        )
        return df_result    
        
    def get_committees_per_conference_country_year_count(
        self,
        conferences: Optional[list[mpq.Conference]]=None,
        years: Optional[list[str]]=None
    ) -> pl.DataFrame:
        """
        Gets Committees counted by conference, country and year

        Args:
            conferences (Optional[list[Conference]], optional): Conferences to filter by. Defaults to None.
            years (Optional[list[str]], optional): Years to filter by. Defaults to None.

        Returns:
            pl.DataFrame: Aggregated results
        """
        
        
        df_query = self.query_client.get_conference_committees(conferences)
        df_result = (
            df_query
            # .filter(pl.col("source_predominant_continent").is_in(conferences))
            .unnest("committee")
            .filter(
                pl.when(years is not None)
                .then(pl.col("year").is_in(years))
                .otherwise(True)       
            )                
            .group_by("conference", "year", "committee_country")
            .agg([
                pl.len().alias("committee_count")
            ])
            .sort("conference", "year", "committee_count", descending=True)
        )
        
        # Ensure that there is no year that exists for a conference and continent combination but 
        # not for other. If it does not exist, it is filled with 0
        df_result_complete = (
            df_result.select("conference").unique()
            .join(df_result.select("year").unique(), how="cross")
            .join(df_result.select("committee_country").unique(), how="cross")
            .join(df_result, on=["conference", "year", "committee_country"], how="left")
            .with_columns(
                pl.col("committee_count").fill_null(0)
            )
            .sort("conference", "year", "committee_count", descending=True)
        )           
        
        return df_result_complete

    def get_committees_per_continent_year_count(
        self,
        conferences: Optional[list[mpq.Conference]]=None, 
        continents: Optional[list[Continent]]=None,
        years: Optional[list[str]]=None,
    ) -> pl.DataFrame:
        """
        Gets Committees counted by conference, continent and year

        Args:
            conferences (Optional[list[Conference]], optional): Conferences to filter by. Defaults to None.
            continents (Optional[list[Continent]], optional): Continents to filter by. Defaults to None.
            years (Optional[list[str]], optional): Years to filter by. Defaults to None.

        Returns:
            pl.DataFrame: Aggregated results
        """
        df_query = self.get_committees_per_conference_country_year_count(
            conferences=conferences
        )
        
        df_result = (
            df_query
            .with_columns([
                pl.col("committee_country")
                .map_elements(self.__get_continent, return_dtype=pl.String)
                .alias("continent")
            ])
            .filter(
                pl.when(continents is not None)
                .then(pl.col("continent").is_in(continents))
                .otherwise(True) &   
                pl.when(years is not None)
                .then(pl.col("year").is_in(years))
                .otherwise(True)                     
            )        
            .group_by("conference", "year", "continent")
            .agg([
                pl.sum("committee_count").alias("committee_count")
            ])
            # .sort("conference", "year", "committee_count", descending=True)        
        )
        
        # Ensure that there is no year that exists for a conference and continent combination but 
        # not for other. If it does not exist, it is filled with 0
        df_result_complete = (
            df_result.select("conference").unique()
            .join(df_result.select("year").unique(), how="cross")
            .join(df_result.select("continent").unique(), how="cross")
            .join(df_result, on=["conference", "year", "continent"], how="left")
            .with_columns(
                pl.col("committee_count").fill_null(0)
            )
            .sort("conference", "year", "committee_count", descending=True)
        )        
 
        return df_result_complete

    def get_committees_per_continent_count(
        self,
        conferences: Optional[list[mpq.Conference]]=None, 
        continents: Optional[list[Continent]]=None,    
    ) -> pl.DataFrame:
        """
        Gets Committees counted by conference and continent

        Args:
            conferences (Optional[list[Conference]], optional): Conferences to filter by. Defaults to None.
            continents (Optional[list[Continent]], optional): Continents to filter by. Defaults to None.
            years (Optional[list[str]], optional): Years to filter by. Defaults to None.

        Returns:
            pl.DataFrame: Aggregated results
        """    
        
        df_query = self.get_committees_per_continent_year_count(
            conferences=conferences,
            continents=continents,
        )
        
        df_result = (
            df_query
            .group_by("conference", "continent")
            .agg([
                pl.sum("committee_count").alias("committee_count")
            ])
            .sort("conference", "continent", "committee_count", descending=True)   
        )
        return df_result
    
    def __get_continent(self, country_code: str) -> Continent:
        """
        Gets continent name for given country

        Args:
            country_code (str): country

        Returns:
            Continent: Continent name
        """
        try:
            continent_code = pc.country_alpha2_to_continent_code(country_code)
            continent_name = pc.convert_continent_code_to_continent_name(continent_code)
            return continent_name
        except KeyError:
            return "Unknown"