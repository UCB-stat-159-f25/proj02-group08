# proj02
Template repository for Project 2, Stat 159/259 Fall 2025

## 01. Environment Setup

- Create the environment called `sotu`
```bash
(base)$ conda env create -f environment.yml
```

- Check the available environments includes `sotu`
```bash
(base)> conda env list

...
sotu                     /home/jovyan/.local/share/envs/sotu
...
```

- Activate the environment, when you need to work 
```bash
(base)> conda activate sotu

(sotu)> 
```

- Deactivate the environment, when you are done working
```bash
(sotu)> conda deactivate
(base)>
```

- Create the `sotu` kernel
```bash

```


## 02. Notebook Management

- pairing .ipynb with .py (for easy diffing and merging)
```bash
(sotu)> jupytext --set-formats ipynb,py:percent testing123.ipynb
[jupytext] Reading testing123.ipynb in format ipynb
[jupytext] Updating notebook metadata with '{"jupytext": {"formats": "ipynb,py:percent"}}'
[jupytext] Updating testing123.ipynb
[jupytext] Updating testing123.py
```
- sync by .ipynb 
```bash
(sotu)> jupytext --sync testing123.ipynb
[jupytext] Reading testing123.ipynb in format ipynb
[jupytext] Loading testing123.py
[jupytext] Unchanged testing123.ipynb
[jupytext] Unchanged testing123.py
```

- sync by .py
```bash
(sotu)> jupytext --sync testing123.py
[jupytext] Reading testing123.py in format py
[jupytext] Loading testing123.ipynb
[jupytext] Unchanged testing123.ipynb
[jupytext] Unchanged testing123.py
```
