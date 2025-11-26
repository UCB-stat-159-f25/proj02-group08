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
# # Part 3: Advanced Text Processing - LDA and BERTopic Topic Modeling

# %%
from spacy import displacy
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis
import pyLDAvis.gensim_models

# %%
# from pathlib import Path 
import pathlib as pl

HOME                = pl.Path.home()
PROJECTS            = "Projects"
STAT                = "STAT_159_259"
PROJ_GRP            = "proj02-group08"

DIR_CODE            = HOME / PROJECTS / STAT / PROJ_GRP / "code"
DIR_DATA            = HOME / PROJECTS / STAT / PROJ_GRP / "data" 
DIR_DATA_RAW        = DIR_DATA / "00_raw"
DIR_DATA_PROCESSED  = DIR_DATA / "01_processed"
DIR_DATA_VECTORIZED = DIR_DATA / "02_vectorized"
DIR_DOCS            = HOME / PROJECTS / STAT / PROJ_GRP / "docs"
DIR_MISC            = HOME / PROJECTS / STAT / PROJ_GRP / "misc"
DIR_NOTEBOOKS       = HOME / PROJECTS / STAT / PROJ_GRP / "notebooks"
DIR_OUTPUTS         = HOME / PROJECTS / STAT / PROJ_GRP / "outputs"
DIR_TESTS           = HOME / PROJECTS / STAT / PROJ_GRP / "tests"

# %%
#from nlp-P02
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# sou = pd.read_csv(
# Path.home() / "proj02-group08" / "data" / "00_raw" / "SOTU.csv"
# )
sou = pd.read_csv(DIR_DATA_RAW / "SOTU.csv")

plt.style.use('seaborn-v0_8-dark')

# %%
#from nlp-P02
import spacy
from tqdm import tqdm
from collections import Counter

nlp = spacy.load("en_core_web_sm")

# %%
# save plots & figures in outputs folder
output_dir = DIR_OUTPUTS


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
def preprocess_text(text): 
    doc = nlp(text) 
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space and len(token.lemma_) > 3]



# %%
from tqdm.auto import tqdm
tqdm.pandas()

# Process all texts - note this takes ~ 5 minutes to run
# processed_docs = sou['Text'].apply(preprocess_text)

processed_docs = sou['Text'].progress_apply(preprocess_text)

# %% [markdown]
# To train an LDA model, use the LdaModel function that we imported a couple of cells back. The last resource linked under the LDA section is especially useful for walking through the steps we have below. *Note: one of the arguments to the LdaModel function is `random_state` which specifies the random seed for reproducibility. Please set yours to 42. Further, the last resource provided uses `LdaMulticore` which is essentially a parallelizable version of our function `LdaModel`. Use `LdaModel` instead, but the usage will be similar, except you can ignore the `iterations` and `workers` arguments..*.

# %%
# Build dictionary from processed_docs, which is a list of tokens extracted from our speeches
dictionary = Dictionary(processed_docs)

corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# %%
# train LDA model with 18 topics
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=18,
    random_state=42
)

# %%
# print the top 10 words for each topic
for topic_id in range (lda_model.num_topics):
    print(f"Topic {topic_id}")
    print(lda_model.print_topic(topic_id, topn=10))
    print("__" * 30)

# %%
# print the topic distribution for the first speech
first_doc_bow = corpus[0]
first_doc_topics = lda_model.get_document_topics(first_doc_bow)

for topic_id, prob in first_doc_topics:
    print(f"Topic {topic_id}: {prob:.4f}")

# %%
# make a visualization using pyLDAvis
pyLDAvis.enable_notebook()

lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(lda_vis, str(output_dir / "lda_vis.html"))
lda_vis

# %% [markdown]
# ### BERTopic
#
# - Train a BERTopic model with a `min_topic_size` of 3 *Hint: use `BERTopic` to instantiate the model and specify `min_topic_size` in here. Actually fit the model using `fit_transform`, which `docs` passed into this.*
# - Output the top 10 words for each topic. 
# - Output the topic distribution for the first speech
# - Make a visualization of the topics (see topic_model.visualize_topics())

# %%
docs = sou['Text'].to_list()

# %%
# train the model - this takes about 30 seconds
topic_model = BERTopic(min_topic_size=3)
topics, probs = topic_model.fit_transform(docs)

# remove stop words from the topics (Hint: use CountVectorizer and then .update_topics on topic_model)
vectorizer_model = CountVectorizer(stop_words="english")
topic_model.update_topics(docs, topics=topics, vectorizer_model=vectorizer_model)

# %%
# output the top 10 words for each topic - hint see get_topic_info
topic_info = topic_model.get_topic_info()

for topic_id in topic_info["Topic"]:
    if topic_id == -1:
        continue
    print(f"Topic {topic_id}")
    print(topic_model.get_topic(topic_id)[:10])  # list of (word, weight) for top 10
    print("__" * 30)

# %%
# output the topic distribution for the first speech
# hint: check out approximate_distribution() and visualize_distribution()
probs_first = topic_model.approximate_distribution([docs[0]])[0]
probs_first = probs_first.flatten()   # ← FIX: makes it 1D properly

for topic_id, prob in enumerate(probs_first):
    if prob > 0:
        print(f"Topic {topic_id}: {prob:.4f}")
dist_fig = topic_model.visualize_distribution(probs_first)
dist_fig.write_html(str(output_dir / "bertopic_distribution_first_doc.html"))

topic_model.visualize_distribution(probs_first)

# %%
# run this cell to visualize the topics
topics_fig = topic_model.visualize_topics()
topics_fig.write_html(str(output_dir / "bertopic_topics.html"))
topic_model.visualize_topics()

# %% [markdown]
# This notebook takes the SOTU speeches and runs both LDA and BERTopic to uncover what presidents actually talk about. LDA shows that almost every topic looks the same. Presidents constantly repeat words like government, states, country, people, nation, etc..., meaning the speeches are super formulaic. BERTopic pulls out more meaningful clusters, like themes around war, economy, and public programs, giving a clearer sense of how different eras emphasize different issues.As we move into the 20th century and beyond, new clusters ermege around modern economic programs, federal agencies, employment, and global diplomacy. More recent speeches incoroprate vocabulary linked to technological progres, economic globalization, and the idea of "Americans" as a collective identity. These shifting clusters reflect changes in national concerns and the expanding scope of presidential responsibilities. Overall, while the fundamental language of presidents remain consistent, the themes of SOTU speeches adapts to the political, economic and global context of each era.

# %%
