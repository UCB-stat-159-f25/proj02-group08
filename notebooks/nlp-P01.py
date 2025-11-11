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
# ## Part 1: Data Loading and Initial Exploration (15 pts)
#
# The data for this project is stored in the `data` folder in your repositories, in the `SOTU.csv` file. The data file is structured as a CSV with columns for president name, speech text, year, and word count in the speech. 
#
# In this section you will:
#
# 1. Import the data into a pandas dataframe
# 2. Perform exploratory data analysis (EDA) including specifically:
# - Analyze the number of speeches per president
# - Analyze the number of speeches per year
# - Analyze the word count distribution
# - Analyze the word count distribution accross years using a rug plot
# - Analyze the average word count per president
# 3. Write commentary on your findings
#
# First, create the `conda` environment with the provided yaml file. Note, it's not unusual for it to take ~15 minutes for the environment to fully install.

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
# ### Exploratory Data Analysis
#
# Replicate the plots below using the hints specified. For each plot, provide some commentary describing the results/anything interesting you might see.

# %% [markdown]
# #### Number of Speeches per President

# %%
# Hint - use value_counts() on the President column
# Hint - sort in order of dataframe

# %%
# Plot 
# Hint - use the .plot() method for Pandas Series, make sure all presidents show up on x-axis

# %% [markdown]
# #### Number of Speeches per Year

# %%
# Hint - Use value counts and sort by years

# %% [markdown]
# #### Word Count Distribution 

# %%
# Hint - try seaborn.histplot()

# %% [markdown]
# #### Word Count Distribution over Year

# %%
# Hint: try seaborn.rugplot()

# %% [markdown]
# #### Word Count Distribution per President

# %%
# Hint: use pandas groupby to get mean word count per president then sort by order
