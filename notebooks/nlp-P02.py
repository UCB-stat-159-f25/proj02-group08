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
# # Part 2: Simple Text Processing - Tokenization, Lemmatization, Word Frequency, Vectorization

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
# sou = pd.read_csv(
#     Path.home() / "proj02-group08" / "data" / "00_raw" / "SOTU.csv"
# )
sou = pd.read_csv(DIR_DATA_RAW / "SOTU.csv")


# %%
sou


# %% [markdown]
# ### Processing Speeches with SpaCy

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
nlp_outputs = [
    nlp(txt)
    for txt in tqdm(after_2000["Text"], desc="Analyzing speeches (>= 2000) with spaCy")
]


# %% [markdown]
# ### Analyze Tokens vs Lemmas

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

token_count = Counter(tokens_filtered)

top_20 = sorted(token_count.items(), key=lambda x: x[1], reverse=True)[:20]
top_20



# %% [markdown]
# #### Lemma List

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

# %% [markdown]
# In the token counts, “year” (406) and “years” (439) appear as two separate items, but the lemma “year” merges them into 845 total occurrences, making it the most frequent concept overall. Lemmatization also highlights words that would otherwise get overlooked. For example, “child” doesn’t appear in the top token list at all, yet the lemma “child” shows up 332 times once forms like “children” and “child’s” are mapped back to the same base word. Overall, lemmatization gives a clearer sense of the underlying topics in the speeches by grouping variations of the same word, instead of treating them as separate terms.

# %% [markdown]
# ### Common Words
#
# #### Common Words per Year Function

# %%
def get_most_common_words(df, year, n=25):
    """
    Processes the SOTU speech for a given year and returns
    the most common non-stopword/punctuation lemmas.
    
    inputs: 
    - df raw unprocessed sou dataframe
    - year
    - n
    outputs: 
    - top n words for that years
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
get_most_common_words(sou, 2024, n=10)

# %% [markdown]
# #### Compare 2023 to 2017
#

# %%
words_2023 = get_most_common_words(sou, 2023, n=20)
words_2017 = get_most_common_words(sou, 2017, n=20)

# %%
words_2023

# %%
words_2017

# %%
df_2017 = pd.DataFrame(words_2017, columns=["word", "count"])
df_2023 = pd.DataFrame(words_2023, columns=["word", "count"])
df_2023.head(), df_2017.head()


# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

datasets = [(df_2017, '2017'), (df_2023, '2023')]

for ax, (dataset, year_str) in zip(axes, datasets):
    sns.barplot(data=dataset, x='word', y='count', ax=ax, color='steelblue')
    ax.set_title(f'{year_str} State of the Union Most Frequent Words', fontsize=13)
    ax.set_xlabel('Word', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(DIR_OUTPUTS / "top_words_2017_2023.png", dpi=300)
plt.show()

# %% [markdown]
# ### TF-IDF Vectorization

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# %% [markdown]
# #### Train the Vectorizer and Transform the Data

# %%
raw_docs = sou["Text"].to_list()

# %%
tfidf_model = TfidfVectorizer()
speech_vectors = tfidf_model.fit_transform(raw_docs)

print(f"Vectorized documents shape: {speech_vectors.shape}")

# %% [markdown]
# #### Plot Speeches
#

# %%
pca = PCA(n_components=2, random_state=42)

df2d = pd.DataFrame(
    pca.fit_transform(speech_vectors.toarray()), 
    columns=['Principle Component 1', 'Principle Component 2']
)

ax = df2d.plot(kind='scatter', x='Principle Component 1', y='Principle Component 2', 
          figsize=(8, 6), s=30)
plt.title('Plot of Vectorized Speeches Principle Components', fontsize=14)
plt.tight_layout()
plt.savefig(DIR_OUTPUTS / "vectorized_speeches_scatterplot_PCA.png", dpi=300)
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
plt.savefig(DIR_OUTPUTS / "vectorized_speeches_heatmap.png", dpi=300)
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
word_nums = [tfidf_model.vocabulary_[word] for word in word_list]

# %%
idf_score = [tfidf_model.idf_[idx] for idx in word_nums] 

# %%
first_speech_vector = speech_vectors[0].toarray().flatten()
tf_idf = [first_speech_vector[idx] for idx in word_nums] 

# %%
pd.DataFrame({"Word": word_list, "IDF Score": idf_score, "TF-IDF Score": tf_idf})
