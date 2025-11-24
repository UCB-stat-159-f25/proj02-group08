#!/usr/bin/env python3

###
# Base Imports
###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###
# Types
###
import typing as ty

###
# Paths
###
 
# from pathlib import Path 
import pathlib as pl

HOME                   = pl.Path.home()
PROJECTS               = "Projects"
STAT                   = "STAT_159_259"
PROJ_GRP               = "proj02-group08"

DIR_CODE               = HOME / PROJECTS / STAT / PROJ_GRP / "code"

DIR_DATA               = HOME / PROJECTS / STAT / PROJ_GRP / "data" 
DIR_DATA_00_RAW        = DIR_DATA / "00_raw"
DIR_DATA_01_PROCESSED  = DIR_DATA / "01_processed"
DIR_DATA_02_VECTORIZED = DIR_DATA / "02_vectorized"
DIR_DATA_03_LDA_BERT   = DIR_DATA / "03_processed_lda_bert"

DIR_DOCS               = HOME / PROJECTS / STAT / PROJ_GRP / "docs"

DIR_MISC               = HOME / PROJECTS / STAT / PROJ_GRP / "misc"

DIR_NOTEBOOKS          = HOME / PROJECTS / STAT / PROJ_GRP / "notebooks"

DIR_OUTPUTS            = HOME / PROJECTS / STAT / PROJ_GRP / "outputs"

DIR_TESTS              = HOME / PROJECTS / STAT / PROJ_GRP / "tests"

CSV_SOTU = "SOTU.csv"

###
# Plots
###

PLOT_STYLE_SEABORN="seaborn-v0_8-dark"

def plot_style(
    style: str="seaborn-v0_8-dark",
):
    plt.style.use(style)


XDIMENSION = 12
YDIMENSION = 8

def plot_settings_for_figure_size(
    xdimension:int=12,
    ydimension:int=8,
):
    # Method 1: Using sns.set()
    sns.set(rc = {'figure.figsize':(xdimension, ydimension)})  # Global setting for all plots
    
    # Method 2: Using Matplotlib's rcParams
    plt.rcParams["figure.figsize"] = (xdimension, ydimension)  # Another global setting
