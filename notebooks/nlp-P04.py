# ---
# jupyter:
#   jupytext:
#     formats: ipynb:percent,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: IPython - sotu
#     language: python
#     name: sotu
# ---

# %% [markdown]
# # Part 4: Choose your own advecnture! Dynamic Topic Modeling

# %%
####################
## CALL THIS BLOCK FROM part00_utils_visuals.py
####################

from bertopic import BERTopic

# read in SOTU.csv using pandas, name the variable `sou` for simplicity
# the below cell is what the output should look like

from src import part00_utils_visuals as part00
# import src.part00_utils_visuals as part00

from src import part01

part00.plot_style(style=part00.PLOT_STYLE_SEABORN)


sou        = part00.pd.read_csv(part00.DIR_DATA_00_RAW / part00.CSV_SOTU)
sou

# %%
timestamps = sou.Year.to_list()
timestamps;

# %%
####################
## MOVE THIS BLOCK TO part03.py
####################

####################
## CALL THIS BLOCK TO part03.py
####################

import spacy
from tqdm import tqdm
from collections import Counter

spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text): 
    doc = nlp(text) 
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space and len(token.lemma_) > 3]


# %%
### Block Output
# 0      [speak, president, present, prepared, remark, ...
# 1      [president, speaker, point, president, turn, f...
# 2      [president, thank, thank, thank, madam, speake...
# 3      [president, thank, thank, thank, good, mitch, ...
# 4      [president, thank, thank, thank, madam, speake...
###

# Process all texts - note this takes ~ 5 minutes to run
# processed_docs = sou['Text'].apply(preprocess_text)

from tqdm.auto import tqdm
tqdm.pandas()  # registers .progress_apply()

# docs       = sou.Text.progress_apply(preprocess_text) #.to_list()

# %%
### Block Output
# 0      [speak, president, present, prepared, remark, ...
# 1      [president, speaker, point, president, turn, f...
# 2      [president, thank, thank, thank, madam, speake...
# 3      [president, thank, thank, thank, good, mitch, ...
# 4      [president, thank, thank, thank, madam, speake...
###


docs_from_part03_csv = part00.DIR_DATA_03_LDA_BERT / "step00_processed_docs.csv"

import ast

docs_from_part03_csv = part00.pd.read_csv(part00.DIR_DATA_03_LDA_BERT / "step00_processed_docs.csv")
docs_from_part03_csv = docs_from_part03_csv.Text
# docs_from_part03_csv = docs_from_part03_csv.apply(ast.literal_eval)
docs_from_part03_csv

# %% [markdown]
# ## Topic evolution over time 
# **Reference:**
# - https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html#visualization

# %%
# train the model - this takes about 30 seconds
MIN_TOPIC_SIZE = 3
IS_VERBOSE = True

step07_topic_model    = BERTopic(min_topic_size=MIN_TOPIC_SIZE, verbose=IS_VERBOSE)
# topics, probabilities = step07_topic_model.fit_transform(docs)
topics, probabilities = step07_topic_model.fit_transform(docs_from_part03_csv.to_list())
topics, probabilities;

# %%
NR_BINS      = 15
TOP_N_TOPICS = 10

###
# help(step07_topic_model.topics_over_time)
###

# topics_over_time(
#     docs: List[str],
#     timestamps: Union[List[str], List[int]],
#     topics: List[int] = None,
#     nr_bins: int = None,
#     datetime_format: str = None,
#     evolution_tuning: bool = True,
#     global_tuning: bool = True
# ) -> pandas.core.frame.DataFrame method of bertopic._bertopic.BERTopic instance
#     Create topics over time.


topics_over_time = step07_topic_model.topics_over_time(docs_from_part03_csv, timestamps, nr_bins=NR_BINS)
topics_over_time

# %%
TOP_N_TOPICS = 10

step07_topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=TOP_N_TOPICS)
