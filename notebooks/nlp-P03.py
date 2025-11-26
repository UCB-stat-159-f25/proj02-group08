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
# # Project 2  Part 3: Advanced Text Processing - LDA and BERTopic Topic Modeling (20 pts)
#

# %% [markdown]
# **Resources:**
# - LDA:
#     - https://medium.com/sayahfares19/text-analysis-topic-modelling-with-spacy-gensim-4cd92ef06e06 
#     - https://www.kaggle.com/code/faressayah/text-analysis-topic-modeling-with-spacy-gensim#%F0%9F%93%9A-Topic-Modeling (code for previous post)
#     - https://towardsdatascience.com/topic-modelling-in-python-with-spacy-and-gensim-dc8f7748bdbf/ 
# - BERTopic:
#     - https://maartengr.github.io/BERTopic/getting_started/visualization/visualize_documents.html#visualize-documents-with-plotly 
#     - https://maartengr.github.io/BERTopic/getting_started/visualization/visualize_topics.html 
#

# %%
# TODO: pass through notebook again remove all stop words, alphanumerics, numbers, punctuations, and other non-essentials

# %%
####################
## MOVE THIS BLOCK TO part03.py
####################

####################
## CALL THIS BLOCK TO part03.py
####################

# from tqdm.auto import tqdm
from spacy import displacy
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis
import pyLDAvis.gensim_models

# %%
####################
## CALL THIS BLOCK FROM part00_utils_visuals.py
####################

# read in SOTU.csv using pandas, name the variable `sou` for simplicity
# the below cell is what the output should look like

from src import part00_utils_visuals as part00
# import src.part00_utils_visuals as part00

from src import part01

part00.plot_style(style=part00.PLOT_STYLE_SEABORN)

sou = part00.pd.read_csv(part00.DIR_DATA_00_RAW / part00.CSV_SOTU)

# %%
sou

# %% [markdown]
# ### LDA
#
# - Train an LDA model with 18 topics
# - Output the top 10 words for each topic. 
# - Output the topic distribution for the first speech
# - Make a visualization

# %% [markdown]
# You may use the next two cells to process the data.

# %%
import spacy
from tqdm import tqdm
from collections import Counter

spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")


# %%
####################
## MOVE THIS BLOCK TO part03.py
####################

####################
## CALL THIS BLOCK TO part03.py
####################

def preprocess_text(text): 
    doc = nlp(text) 
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space and len(token.lemma_) > 3]


# %%
# Process all texts - note this takes ~ 5 minutes to run
# processed_docs = sou['Text'].apply(preprocess_text)

from tqdm.auto import tqdm
tqdm.pandas()  # registers .progress_apply()

step00_processed_docs = sou['Text'].progress_apply(preprocess_text)


# %% [markdown]
# Example Progress Bar: ![ProgressBar_Screenshot 2025-11-23 at 15.05.05.png](attachment:d9bb6faf-2cb4-4593-8ec2-24cfc2da4fb0.png))

# %%
type(step00_processed_docs), step00_processed_docs

# %%
part01.save_the_processed_data_to_csv(data=step00_processed_docs, filepath=part00.DIR_DATA_03_LDA_BERT / "step00_processed_docs.csv")

# %% [markdown]
# To train an LDA model, use the LdaModel function that we imported a couple of cells back. The last resource linked under the LDA section is especially useful for walking through the steps we have below. *Note: one of the arguments to the LdaModel function is `random_state` which specifies the random seed for reproducibility. Please set yours to 42. Further, the last resource provided uses `LdaMulticore` which is essentially a parallelizable version of our function `LdaModel`. Use `LdaModel` instead, but the usage will be similar, except you can ignore the `iterations` and `workers` arguments..*.

# %%
# processed_docs = read_csv(...)

import ast

step01_processed_docs_from_csv = part00.pd.read_csv(part00.DIR_DATA_03_LDA_BERT / "step00_processed_docs.csv")
step01_processed_docs_from_csv = step01_processed_docs_from_csv["Text"]
step01_processed_docs_from_csv = step01_processed_docs_from_csv.apply(ast.literal_eval)
type(step01_processed_docs_from_csv), step01_processed_docs_from_csv

# %%
# Build dictionary from processed_docs, which is a list of tokens extracted from our speeches
step02_build_dict_from_processed_docs = Dictionary(step01_processed_docs_from_csv)
step02_build_dict_from_processed_docs;

# %%
step03_corpus = [doc for doc in step01_processed_docs_from_csv]
step03_corpus = [step02_build_dict_from_processed_docs.doc2bow(doc) for doc in step01_processed_docs_from_csv]
step03_corpus;

# %%
# train LDA model with 18 topics

NUM_OF_TOPICS      = 18
RANDOM_SEED_NUM    = 42
NUM_OF_PASSES      = 10
PROGRESS_FREQUENCY = 0 # batch learning

lda_model = LdaModel(
    corpus=step03_corpus,
    id2word=step02_build_dict_from_processed_docs,
    num_topics=NUM_OF_TOPICS,
    random_state=RANDOM_SEED_NUM,
    passes=NUM_OF_PASSES,
    update_every=PROGRESS_FREQUENCY,
)
lda_model

# %%
# print the top 10 words for each topic
NUM_OF_TOP_N_TOPICS = 10

print(f"--- Top LDA topics. ---")
for idx, topic in lda_model.print_topics(num_words=NUM_OF_TOP_N_TOPICS):
    print(f"Topic: {idx} \nWords: {topic}\n")

# %%
# lda_model.print_topics(-1)

# %%
# print the topic distribution for the first speech
# Where does a text belong to

# step04_topic_dist_first_speech = lda_model[step03_corpus[0]]
step04_topic_dist_first_speech = lda_model[step03_corpus][0]
step04_topic_dist_first_speech

# %%
# make a visualization using pyLDAvis
pyLDAvis.enable_notebook()

# %%
pyLDAvis.gensim_models.prepare(lda_model, step03_corpus, step02_build_dict_from_processed_docs)

# %%
# print the topic distribution for the first speech

SPEECH_ID = 0

step05_first_speech_bow = step03_corpus[SPEECH_ID]
step05_first_speech_bow;

step06_first_speech_topics = lda_model.get_document_topics(bow=step05_first_speech_bow)

for topic_id, prob in step06_first_speech_topics:
    print(f"Topic {topic_id}: {prob}")

# %% [markdown]
# ### BERTopic
#
# - Train a BERTopic model with a `min_topic_size` of 3 *Hint: use `BERTopic` to instantiate the model and specify `min_topic_size` in here. Actually fit the model using `fit_transform`, which `docs` passed into this.*
# - Output the top 10 words for each topic. 
# - Output the topic distribution for the first speech
# - Make a visualization of the topics (see topic_model.visualize_topics())

# %%
docs = sou['Text'].to_list()
docs[0];

# %%
# train the model - this takes about 30 seconds
MIN_TOPIC_SIZE = 3
IS_VERBOSE = True

step07_topic_model    = BERTopic(min_topic_size=MIN_TOPIC_SIZE, verbose=IS_VERBOSE)
topics, probabilities = step07_topic_model.fit_transform(docs)

# %% [markdown]
# ![ProgressBar_BERTopic_Screenshot 2025-11-24 at 17.27.20.png](attachment:5e375b68-7ac9-4f39-9d72-521bbcedbd53.png)

# %%
step07_topic_model.get_topic_info()

# %%
step07_topic_model.get_topic(0)


# %%
# remove stop words from the topics (Hint: use CountVectorizer and then .update_topics on topic_model)

LANG="english"

step08_vectorizer_model = CountVectorizer(stop_words=LANG)
step07_topic_model.update_topics(docs, topics=topics, vectorizer_model=step08_vectorizer_model)


# %%
# output the top 10 words for each topic - hint see get_topic_info

step09_topic_info = step07_topic_model.get_topic_info()
step09_topic_info

# %%
topics, probabilities = step07_topic_model.fit_transform(docs)

# %%
# output the topic distribution for the first speech
# hint: check out approximate_distribution() and visualize_distribution()
###
# references:
# 1. https://maartengr.github.io/BERTopic/getting_started/distribution/distribution.html
#
# topic_distr, _ = topic_model.approximate_distribution(docs)
# topic_model.visualize_distribution(topic_distr[1])
###

CALCULATE_TOKENS = True

topic_distributions, topic_token_distributions = step07_topic_model.approximate_distribution(docs, calculate_tokens=CALCULATE_TOKENS)

# %%
topic_distributions.shape

# %%
DOC_ROW_ID = 0

step07_topic_model.visualize_distribution(topic_distribution[DOC_ROW_ID])

# %%
import ipywidgets as widgets

@widgets.interact(DOC_ROW_ID=(0, len(topic_distribution)-1))
def show(DOC_ROW_ID=0):
    return step07_topic_model.visualize_distribution(topic_distribution[DOC_ROW_ID])


# %%
df = step07_topic_model.visualize_approximate_distribution(docs[DOC_ROW_ID], topic_token_distributions[DOC_ROW_ID])
df

# %%
# run this cell to visualize the topics
step07_topic_model.visualize_topics()

# %%
# TODO: pass through notebook again remove all stop words, alphanumerics, numbers, punctuations, and other non-essentials
