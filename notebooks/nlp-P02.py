# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# # Project 2: Reproducibility in Natural Language Processing
#
#

# %% [markdown]
# ### Read Data

# %%
# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-dark')

# %%
# read in SOTU.csv using pandas, name the variable `sou` for simplicity
# the below cell is what the output should look like


# %%
sou


# %% [markdown]
# ## Part 2: Simple Text Processing - Tokenization, Lemmatization, Word Frequency, Vectorization (20 pts)
#
# Now we will start working on simple text processing using the `SpaCy` package and the same dataset as Part 1. The package should already be included in the `environment.yml`. However, we will also need to download `en_core_web_sm`, an English language text processing model. To do this, while having your `sotu` environment activated, run the following:
#
# ```
# python -m spacy download en_core_web_sm
# ```
#
# Now, you should be good to go!
#
# Some important definitions:
#
# - *Token*: a single word or piece of a word
# - *Lemma*: the core component of a word, e.g., "complete" is the lemma for "completed" and "completely"
# - *Stop Word*: a common word that does not add semantic value, such as "a", "and", "the", etc.
# - *Vectorization*: representing a document as a vector where each index in the vector corresponds to a token or word and each entry is the count.
#
# In this section, we will explore the most common tokens and lemmas throughout different slices of the speech data. We will also develop vectorization representations of the speeches. 
#
#  The core steps are:
#
# 1. Process speeches using the SpaCy nlp module
# 2. Analyze Tokens vs Lemmas:
# - Create a list of all tokens across all speeches that are not stop words, punctuation, or spaces.
# - Create a second list of the lemmas for these same tokens.
# - Display the top 25 for each of these and compare.
# 3. Analyze common word distributions over different years:
# - Create a function that takes the dataset and a year as an input and outputs the top n lemmas for that year's speeches
# - Compare the top 10 words for 2023 versus 2019
# 4. Document Vectorization:
# - Train a Term Frequency-Inverse Document Frequency (TF-IDF) vectorization model using your processed dataset and scikit learn
# - Output the feature vectors 
#
# **Helpful Resources:**
# - https://realpython.com/natural-language-processing-spacy-python/
# - https://www.statology.org/text-preprocessing-feature-engineering-spacy/ 
# - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html# 
# - https://www.geeksforgeeks.org/nlp/how-to-store-a-tfidfvectorizer-for-future-use-in-scikit-learn/ 
#
#

# %% [markdown]
# ### Processing Speeches with SpaCy
#
# Lets study just speeches from 2000 and onwards to begin with. So, be sure to subset your DataFrame to just these speeches before continuing!

# %%
import spacy
from tqdm import tqdm
from collections import Counter

nlp = spacy.load("en_core_web_sm")


# %%
# subset the speech dataframe for speeches from 2000 and onwards

# %%
# Process each speeches using the 'nlp' function
# Hint - loop through each speech and pass the text into the nlp function above, storing the output in a list
# should take about 30 seconds to process

# %% [markdown]
# ### Analyze Tokens vs Lemmas
#
# #### Token List
#
# Create a list of tokens across all speeches that are not spaces, stopwords, or punctuation. Make each token lowercase as well. *Hint: each element of the list we just created are themselves lists of tokens. Token objects have attributes `is_stop`, `is_punct`, and `is_space`.*

# %%

# %%
# print top 20 tokens
# Hint - use Counter, and one of the Counter object's methods to display the top 20

# %% [markdown]
# #### Lemma List
#
# Do the same as above, but for lemmas. *Hint: recall lemmas are components of words. Each token should have an attribute to extract the lemma.*

# %%

# %% [markdown]
# #### Token versus Lemma Comparison
#
# What do you notice about the top tokens versus the top lemmas? 
# Consider two tokens - "year" and "years" - how do their counts compare to the lemma "year"?
# What about the lemma "child"?

# %% [markdown]
# ### Common Words
#
# #### Common Words per Year Function
#
# Fill in the below function to obtain the n-most common words in speeches for a given year.
#
# inputs: 
# - df raw unprocessed sou dataframe
# - year
# - n
# outputs: 
# - top n words for that years
#
# steps:
# - subset the dataframe for the year of interest - note the years might not be in int type
# - process the subsetted dataframe with spacy
# - get the lemmas across all those speeches
# - count the top n lemmas

# %%
def get_most_common_words(df, year, n=25):
    """
    Processes the SOTU speech for a given year and returns
    the most common non-stopword/punctuation lemmas.
    """

    # Step 1: Subset df

    # Step 2: Process the text with spaCy
    
    # Step 3: Get lemmas
    
    return ...


# %%
# test it on 2024

# %% [markdown]
# #### Compare 2023 to 2017
#
# Run your function from the previous step to get the top 20 words for 2017 and 2023. Plot the words and their frequencies in a barchart and replicate the figure below.

# %%
words_2023 = ...
words_2017 = ...

# %%
words_2023

# %%
words_2017

# %%
# Hint - put the words and counts into a pd Dataframe for better structure
# and to make plotting easier
df_2017 = ...
df_2023 = ...

# %%
# Hint - use seaborn, subplots, and rotate tick labels

# %% [markdown]
# ### TF-IDF Vectorization
#
# To use statsitical alorithms on documents, we need to transform them into vectors, where each element of the vector corresponds to a particular word in a document or corpus of documents. One common way is via TF-IDF embeddings. LLMs work similarly - they typically use transformer models to generate text embeddings before sending text through a deep neural network. 
#
# Here we will fit a TF-IDF vectorizer, plot all the speeches on a 2-D grid using PCA and also using a heatmap, and examine TF-IDF scores for the top 10 most common words in the first speech. This is a good resource here: https://medium.com/GeoffreyGordonAshbrook/vector-visualization-2d-plot-your-tf-idf-with-pca-83fa9fccb1d 

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# %% [markdown]
# #### Train the Vectorizer and Transform the Data

# %%
# you may use this as input to fit the TF-IDF vectorizer
raw_docs = sou["Text"].to_list()

# %%
# Hint - use fit_transform for vectorizer and PCA

# %% [markdown]
# The output of `fit_transform()` will be a matrix where each row corresponds to a speech, each column corresponds to a word in the corpus of speeches, and the value is the TF-IDF score which measures the importance of that word in that speech, relative to the rest of the speeches.

# %% [markdown]
# #### Plot Speeches
#
# - First used PCA to generate the first chart
# - Second use seaborn heatmap with a log-scaled color axis to generate the second chart

# %%
# Step 1: Set PCA to find first 2 principal components

# Step 2: Create a new dataframe where each row is a speech, and each column is a projection onto
# one of the two principal components

# Plot Data Visualization (Matplotlib)

# %%
# Hint - vectorized_docs is a sparse matrix whose rows are speeches and columns are tokens, with each
# value being a TF-IDF score. Densify this array first, and then plot using seaborn.

# %% [markdown]
# #### Get the TF-IDF value for certain words and documents

# %%
word_list = ['year',
 'america',
 'people',
 'american',
 'work',
 'new',
 'job',
 'country',
 'americans',
 'world'] # top ten most common words through whole corpus

# %%
word_nums = ... # get each word's index number using the .vocabular_ attributed of vectorizer

# %%
idf_score = ... # get their IDF score by using .idf_ at the indices from the previous step

# %%
tf_idf = ... # get the tf_idf score for the first speech

# %%
pd.DataFrame({"Word": word_list, "IDF Score": idf_score, "TF-IDF Score": tf_idf})
