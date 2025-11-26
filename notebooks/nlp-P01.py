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
# # Project 2 - Part 1: Data Loading and Initial Exploration (15 pts)

# %% [markdown]
# ### Read Data

# %%
# imports
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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
import os
os.getcwd()

# %%
# read in SOTU.csv using pandas, name the variable `sou` for simplicity
# example from class df = pd.read_csv(Path.home()/"shared/climate-data/monthly_in_situ_co2_mlo_cleaned.csv")
#pd.read_csv(Path.home()/"shared/climate-data/monthly_in_situ_co2_mlo_cleaned.csv")
# sou = pd.read_csv(Path.home() / "proj02-group08" / "data" / "00_raw" / "SOTU.csv")
sou = pd.read_csv(DIR_DATA_RAW / "SOTU.csv")



# show first few rows
sou.head()


# %% [markdown]
# ### Exploratory Data Analysis
#
# Replicate the plots below using the hints specified. For each plot, provide some commentary describing the results/anything interesting you might see.

# %% [markdown]
# #### Number of Speeches per President

# %%
# Hint - use value_counts() on the President column
# Hint - sort in order of dataframe
speeches_per_president_df = sou['President'].value_counts().sort_values(ascending=False).reset_index()
speeches_per_president_df.columns = ['President', 'Number of Speeches']

speeches_per_president_df.head()

# %%
speeches_per_president = sou['President'].value_counts().sort_values(ascending=False)

# Plot
plt.figure(figsize=(12,6))
speeches_per_president.plot(kind='bar')

plt.title("Number of Speeches per President")
plt.xlabel("President")
plt.ylabel("Number of Speeches")
plt.xticks(rotation=70)
plt.show()

plt.savefig(DIR_OUTPUTS / "numberOfSpechesPerPresident.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# #### Number of Speeches per Year

# %%
speeches_per_year = sou['Year'].value_counts().sort_index()

# Display the Series
speeches_per_year

# Plot
plt.figure(figsize=(12,6))
speeches_per_year.plot(kind='line')

plt.title("Number of Speeches per Year")
plt.xlabel("Year")
plt.ylabel("Number of Speeches")
plt.xticks(rotation=70)
plt.show()

plt.savefig(DIR_OUTPUTS / "numberOfSpechesPerPresident.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# #### Word Count Distribution 

# %%
plt.figure(figsize=(10,6))
sns.histplot(sou['Word Count'], kde=True, bins=30, color='blue')
plt.title("Distribution of State Of The Union Speech Counts")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()


plt.savefig("../outputs/DistributionOfState.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# #### Word Count Distribution over Year

# %%
# Hint: try seaborn.rugplot()
plt.figure()
sns.scatterplot(data=sou, x='Word Count', y='Year')
sns.rugplot(data=sou, x='Word Count', y='Year')
plt.title("Speech Year Versus Word Count")
plt.show()


plt.savefig(DIR_OUTPUTS / "numberOfSpechesPerPresident.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# #### Word Count Distribution per President

# %%
avg_word_count = sou.groupby('President')['Word Count'].mean().sort_values(ascending=False)

# display
avg_word_count

# %%
plt.figure(figsize=(12,6))
avg_word_count.plot(kind='bar', color='skyblue')
plt.title("Average Word Count per President")
plt.xlabel("President")
plt.ylabel("Average Word Count")
plt.xticks(rotation=70)
plt.show()


plt.savefig(DIR_OUTPUTS / "numberOfSpechesPerPresident.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# The average word count per president highlights how speech lengths have varied throughout history. Early presidents generally gave shorter addresses, while some nineteenth-century presidents, like James K. Polk, and a few early twentieth-century leaders delivered very long speeches. Modern presidents typically fall in the mid-range, between four and eight thousand words. Overall, speech length does not follow a consistent trend over time, but instead varies by individual president and historical era.

# %% [markdown]
# # Number of Speeches per President
# This chart shows the number of speeches recorded for each U.S. president in the State of the Union dataset. George Washington appears with the highest count because he delivered multiple formal addresses at the beginning of the nation’s history, establishing the tradition of presidential messages to Congress. Early presidents often gave more frequent written or spoken addresses, and the dataset captures these as separate speeches. As a result, Washington naturally rises to the top in total speech count.

# %% [markdown]
# # Number of State of the Union Speechs per Year
# this graph shows the number of State of the Union speeches delivered each year. The trend remains relatively steady overall, but there is a noticeable peak between 1950 and 2000. This increase reflects a period of major political activity—such as the Cold War, civil rights movement, and significant economic transitions—when presidents delivered more frequent and detailed annual messages to Congress. The elevated speech counts during these decades capture how national and global events shaped the intensity of presidential communication.

# %% [markdown]
# # Distribution of State of the Union Speech word counts
# This chart shows the distribution of word counts across all State of the Union speeches. It highlights how speech length varies significantly between presidents and over time. Some speeches are relatively short and concise, while others are much longer, reflecting the context, historical period, and priorities of each administration. Analyzing word count distribution helps identify trends in presidential communication, such as shifts toward more detailed or policy-heavy addresses.

# %% [markdown]
# # Speech Year Versus Word Count
# This graph shows how the word count of State of the Union speeches changes over time. It highlights trends in presidential communication, with some periods featuring shorter, more concise addresses and others marked by longer, more detailed speeches.
