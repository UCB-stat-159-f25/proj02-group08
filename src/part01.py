#!/usr/bin/env python3

import spacy
import pandas as pd
import pathlib as pl

from tqdm import tqdm
from collections import Counter
from typing import Optional, Union

MODEL_LANG_EN = "en_core_web_sm"

# https://spacy.io/models/en
def get_spacy_model(
    name: str = MODEL_LANG_EN
) -> Optional[spacy.Language]: 
    try:
        return spacy.load(name)
    except OSError:
        download(name)
        return spacy.load(name)


def subset_speeches_using_pandas_filter_by_year(
    data_speeches: pd.DataFrame,
    year_start: int = 2000,
    year_end: Optional[int] = None,
) -> pd.DataFrame:
    
    return data_speeches[data_speeches['Year'] >= year_start] 


def process_speeches_using_spacy_nlp(
    data_speeches: pd.DataFrame,
    year_start: int = 2000,
    year_end: Optional[int] = None,
) -> None:
    pass

def save_the_processed_data_to_csv(
    data: Union[pd.DataFrame, pd.Series],
    filepath: Union[str, pl.Path],
) -> None:

    # Convert Series to DataFrame
    if isinstance(data, pd.Series):
        data_to_save: pd.DataFrame = data.to_frame()
    else:
        data_to_save: pd.DataFrame = data

    data_to_save.to_csv(filepath, index=False)


def save_the_vectorized_data_to_csv(
    data_vectorized: 
    filepath: Union[str, pl.Path],
) -> None:
    pass