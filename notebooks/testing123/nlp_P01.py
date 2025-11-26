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
####################
## MOVE THIS BLOCK TO part00_utils_visuals.py
####################

####################
## CALL THIS BLOCK TO part00_utils_visuals.py
####################

# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-dark')


# %%
####################
## MOVE THIS BLOCK TO part00_utils_visuals.py
####################

####################
## CALL THIS BLOCK TO part00_utils_visuals.py
####################

###
# Types
###
import typing as ty

###
# Paths
###
 
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

###
# 
###
XDIMENSION = 12
YDIMENSION = 8

def plot_settings_for_figure_size(
    xdimension:int=12,
    ydimension:int=6
):
    # Method 1: Using sns.set()
    sns.set(rc = {'figure.figsize':(xdimension, ydimension)})  # Global setting for all plots
    
    # Method 2: Using Matplotlib's rcParams
    plt.rcParams["figure.figsize"] = (xdimension, ydimension)  # Another global setting

plot_settings_for_figure_size(XDIMENSION, YDIMENSION)

# %%
####################
## MOVE THIS BLOCK TO part00_utils_visuals.py
####################

####################
## CALL THIS BLOCK TO part00_utils_visuals.py
####################

# read in SOTU.csv using pandas, name the variable `sou` for simplicity
# the below cell is what the output should look like

sou = pd.read_csv(filepath_or_buffer=pl.Path(DIR_DATA_RAW) / "SOTU.csv", header=0)


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

number_speeches_per_president = sou["President"].value_counts(sort=False)
number_speeches_per_president


# %%
####################
## MOVE THIS BLOCK TO part01.py
####################

def plot_number_speeches_per(
    series: pd.Series,
    kind_of_plot: str = "bar",
    title: str = "Enter Title Here",
    xlabel: ty.Optional[str] = None,
    ylabel: ty.Optional[str] = None,
    figsize: ty.Tuple[int, int] = (12, 6)
) -> plt.Figure:

    # plot configuration
    fig, ax = plt.subplots(figsize=figsize)

    # plot series
    series.plot(kind=kind_of_plot)

    # plot labels

    ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.show()


# %%
####################
## CALL THIS BLOCK from part01.py
####################


# Plot 
# Hint - use the .plot() method for Pandas Series, make sure all presidents show up on x-axis

# number_speeches_per_president.plot(kind="bar");
plot_number_speeches_per(
    series=number_speeches_per_president, 
    kind_of_plot="bar",
    title="Number of Speeches per President",
    xlabel="President",
    ylabel="Number of Speeches",
    figsize=(XDIMENSION, YDIMENSION)
)

# %% [markdown]
# #### Number of Speeches per Year

# %%
# Hint - Use value counts and sort by years

number_speeches_per_year = sou["Year"].value_counts(sort=False)
number_speeches_per_year

# %%
####################
## CALL THIS BLOCK from part01.py
####################

plot_number_speeches_per(
    series=number_speeches_per_year, 
    kind_of_plot="line",
    title="Number of Speeches per Year",
    xlabel="President",
    ylabel="Number of Speeches"
)

# %% [markdown]
# #### Word Count Distribution 

# %%
# Hint - try seaborn.histplot()

word_count_distribution = sou["Word Count"]
word_count_distribution

# %%
sns.histplot(word_count_distribution);

# %% [markdown]
# #### Word Count Distribution over Year

# %%
years = sou["Year"]
years


# %%
####################
## MOVE THIS BLOCK TO part01.py
####################

# Hint: try seaborn.rugplot()
# sns.scatterplot(data=sou, x="Word Count", y="Year")
# sns.rugplot(data=sou, x="Word Count", y="Year")

def plot_word_count_distribution_per(
    dataframe: pd.DataFrame,
    title: str = "Enter Title Here",
    xlabel: ty.Optional[str] = None,
    ylabel: ty.Optional[str] = None,
    figsize: ty.Tuple[int, int] = (12, 6)
) -> plt.Figure:

    # plot configuration
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.scatterplot(data=dataframe, x=xlabel, y=ylabel, ax=ax)
    sns.rugplot(data=dataframe, x=xlabel, y=ylabel, ax=ax)

    # plot labels

    ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    # fig.tight_layout()
    # return fig

####################
## CALL THIS BLOCK from part01.py
####################

plot_word_count_distribution_per(
    dataframe=sou,
    title="Speech Year Versus Word Count",
    xlabel="Word Count",
    ylabel="Year",
    figsize=(XDIMENSION, YDIMENSION)
)

# %% [markdown]
# #### Word Count Distribution per President

# %%
# Hint: use pandas groupby to get mean word count per president then sort by order

# %%
word_count_distribution_per_president = sou\
    .groupby(["President"])["Word Count"]\
    .mean()

# %%
word_count_distribution_per_president

# %%
####################
## CALL THIS BLOCK from part01.py
####################

# TODO: adjust this plot. The arrangement is not completely correct.

plot_number_speeches_per(
    series=word_count_distribution_per_president, 
    kind_of_plot="bar",
    title="Average State of the Union Word Count per President",
    xlabel="President",
    ylabel="Average Word Count",
    figsize=(XDIMENSION, YDIMENSION)
)

