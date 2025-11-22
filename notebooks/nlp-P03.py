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
from spacy import displacy
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis
import pyLDAvis.gensim_models


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
# Process all texts - note this takes ~ 5 minutes to run
processed_docs = sou['Text'].apply(preprocess_text)

# %% [markdown]
# To train an LDA model, use the LdaModel function that we imported a couple of cells back. The last resource linked under the LDA section is especially useful for walking through the steps we have below. *Note: one of the arguments to the LdaModel function is `random_state` which specifies the random seed for reproducibility. Please set yours to 42. Further, the last resource provided uses `LdaMulticore` which is essentially a parallelizable version of our function `LdaModel`. Use `LdaModel` instead, but the usage will be similar, except you can ignore the `iterations` and `workers` arguments..*.

# %%
# Build dictionary from processed_docs, which is a list of tokens extracted from our speeches

# %%
# train LDA model with 18 topics

# %%
# print the top 10 words for each topic

# %%
# print the topic distribution for the first speech

# %%
# make a visualization using pyLDAvis
pyLDAvis.enable_notebook()
...

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

# remove stop words from the topics (Hint: use CountVectorizer and then .update_topics on topic_model)

# %%
# output the top 10 words for each topic - hint see get_topic_info

# %%
# output the topic distribution for the first speech
# hint: check out approximate_distribution() and visualize_distribution()

# %%
# run this cell to visualize the topics
topic_model.visualize_topics()
