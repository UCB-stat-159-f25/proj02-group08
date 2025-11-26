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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Project 2 - Part 2: Simple Text Processing - Tokenization, Lemmatization, Word Frequency, Vectorization (20 pts)
#
#
#

# %% [markdown]
# ### Read Data

# %%
# imports
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm 
import numpy as np

plt.style.use('seaborn-v0_8-dark')



# %%
# read in SOTU.csv using pandas, name the variable `sou` for simplicity
# the below cell is what the output should look like
sou = pd.read_csv(
    Path.home() / "proj02-group08" / "data" / "00_raw" / "SOTU.csv"
)


# %%
sou


# %% [markdown]
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
after_2000 = sou[sou["Year"] >= 2000].copy()


# %%
# Process each speeches using the 'nlp' function
# Hint - loop through each speech and pass the text into the nlp function above, storing the output in a list
# should take about 30 seconds to process
nlp_outputs = [
    nlp(txt)
    for txt in tqdm(after_2000["Text"], desc="Analyzing speeches (>= 2000) with spaCy")
]


# %% [markdown]
# ### Analyze Tokens vs Lemmas
#
# #### Token List
#
# Create a list of tokens across all speeches that are not spaces, stopwords, or punctuation. Make each token lowercase as well. *Hint: each element of the list we just created are themselves lists of tokens. Token objects have attributes `is_stop`, `is_punct`, and `is_space`.*

# %%
from collections import Counter

tokens_filtered = []


for doc in nlp_outputs:
    for tok in doc:
        if tok.is_space:
            continue
        if tok.is_stop:
            continue
        if tok.is_punct:
            continue
            
        tokens_filtered.append(tok.text.lower())

# %%
# print top 20 tokens
# Hint - use Counter, and one of the Counter object's methods to display the top 20

token_count = Counter(tokens_filtered)

top_20 = sorted(token_count.items(), key=lambda x: x[1], reverse=True)[:20]
top_20



# %% [markdown]
# #### Lemma List
#
# Do the same as above, but for lemmas. *Hint: recall lemmas are components of words. Each token should have an attribute to extract the lemma.*

# %%
lemmas_filtered = []

for doc in nlp_outputs:
    for tok in doc:
        if tok.is_space:
            continue
        if tok.is_stop:
            continue
        if tok.is_punct:
            continue
            
        lemmas_filtered.append(tok.lemma_.lower())

lemma_count = Counter(lemmas_filtered)

top_20_lemmas = sorted(lemma_count.items(), key=lambda x: x[1], reverse=True)[:20]
top_20_lemmas


# %% [markdown]
# #### Token versus Lemma Comparison
#
# What do you notice about the top tokens versus the top lemmas? 
# Consider two tokens - "year" and "years" - how do their counts compare to the lemma "year"?
# What about the lemma "child"?

# %% [markdown]
# In the token counts, “year” (406) and “years” (439) appear as two separate items, but the lemma “year” merges them into 845 total occurrences, making it the most frequent concept overall. Lemmatization also highlights words that would otherwise get overlooked. For example, “child” doesn’t appear in the top token list at all, yet the lemma “child” shows up 332 times once forms like “children” and “child’s” are mapped back to the same base word. Overall, lemmatization gives a clearer sense of the underlying topics in the speeches by grouping variations of the same word, instead of treating them as separate terms.

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
    filtered_speeches = df[df["Year"].astype(int) == int(year)]
    
    if filtered_speeches.empty:
        return []

    # Step 2: Process the text with spaCy
    extracted_lemmas = []
    for speech_text in tqdm(filtered_speeches['Text'], desc=f"Processing {year}"):
        processed_doc = nlp(speech_text)
        
        # Step 3: Get lemmas
        for token in processed_doc:
            if token.is_space:
                continue
            if token.is_stop:
                continue
            if token.is_punct:
                continue
            extracted_lemmas.append(token.lemma_.lower())
    lemma_counts = Counter(extracted_lemmas)
    return lemma_counts.most_common(n)



# %%
# test it on 2024
get_most_common_words(sou, 2024, n=10)

# %% [markdown]
# #### Compare 2023 to 2017
#
# Run your function from the previous step to get the top 20 words for 2017 and 2023. Plot the words and their frequencies in a barchart and replicate the figure below.

# %%
words_2023 = get_most_common_words(sou, 2023, n=20)
words_2017 = get_most_common_words(sou, 2017, n=20)

# %%
words_2023

# %%
words_2017

# %%
# Hint - put the words and counts into a pd Dataframe for better structure
# and to make plotting easier
df_2017 = pd.DataFrame(words_2017, columns=["word", "count"])
df_2023 = pd.DataFrame(words_2023, columns=["word", "count"])
df_2023.head(), df_2017.head()


# %%
# Hint - use seaborn, subplots, and rotate tick labels
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

datasets = [(df_2017, '2017'), (df_2023, '2023')]

for ax, (dataset, year_str) in zip(axes, datasets):
    sns.barplot(data=dataset, x='word', y='count', ax=ax, color='steelblue')
    ax.set_title(f'{year_str} State of the Union Most Frequent Words', fontsize=13)
    ax.set_xlabel('Word', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("../outputs/top_words_2017_2023.png", dpi=300)
plt.show()

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
tfidf_model = TfidfVectorizer()
speech_vectors = tfidf_model.fit_transform(raw_docs)

print(f"Vectorized documents shape: {speech_vectors.shape}")

# %% [markdown]
# The output of `fit_transform()` will be a matrix where each row corresponds to a speech, each column corresponds to a word in the corpus of speeches, and the value is the TF-IDF score which measures the importance of that word in that speech, relative to the rest of the speeches.

# %% [markdown]
# #### Plot Speeches
#
# - First used PCA to generate the first chart
# - Second use seaborn heatmap with a log-scaled color axis to generate the second chart

# %%
# Step 1: Set PCA to find first 2 principal components
pca = PCA(n_components=2, random_state=42)

# Step 2: Create a new dataframe where each row is a speech, and each column is a projection onto
# one of the two principal components
df2d = pd.DataFrame(
    pca.fit_transform(speech_vectors.toarray()), 
    columns=['Principle Component 1', 'Principle Component 2']
)

# Plot Data Visualization (Matplotlib)
ax = df2d.plot(kind='scatter', x='Principle Component 1', y='Principle Component 2', 
          figsize=(8, 6), s=30)
plt.title('Plot of Vectorized Speeches Principle Components', fontsize=14)
plt.tight_layout()
plt.savefig("../outputs/vectorized_speeches_scatterplot_PCA.png", dpi=300)
plt.show()

# %%
densify_array = speech_vectors.toarray()

plt.figure(figsize=(14, 10))

sns.heatmap(
    densify_array,
    cmap="rocket",  
    norm=LogNorm(),  
    cbar_kws={'label': 'TF-IDF Score'},
    xticklabels=928,  
    yticklabels=10     
)

plt.title("Vectorized Speeches", fontsize=16)
plt.xlabel("Vector Index", fontsize=12)
plt.ylabel("Speech Index", fontsize=12)
plt.tight_layout()
plt.savefig("../outputs/vectorized_speeches_heatmap.png", dpi=300)
plt.show()

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
word_nums = [tfidf_model.vocabulary_[word] for word in word_list] # get each word's index number using the .vocabular_ attributed of vectorizer

# %%
idf_score = [tfidf_model.idf_[idx] for idx in word_nums] # get their IDF score by using .idf_ at the indices from the previous step

# %%
first_speech_vector = speech_vectors[0].toarray().flatten()
tf_idf = [first_speech_vector[idx] for idx in word_nums] # get the tf_idf score for the first speech

# %%
pd.DataFrame({"Word": word_list, "IDF Score": idf_score, "TF-IDF Score": tf_idf})
