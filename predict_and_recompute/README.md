# New communication hiding conjugate gradient variants

This folder contains materials to help reproduce the results of:

    @article{chen_19,
        Author = {Tyler Chen},
        Title = {New communication hiding conjugate gradient variants},
        Howpublished = {In progress},
        Year = {2019}
    }

An extended introduction to this paper can be found [here](http://chen.pw/research/publications/chen_19.html).

## Repository contents

The contents of this folder are roughly structured as follows:

- `experiments` : ipython notebooks and python scripts for generated numerical experiments
    - `data` : raw convergence data and table summary statisitcs
    - `figures` : convergence plots for all numerical experiments, and compiled table summary statistics
- `cg_variants` : implementations of various conjugate gradient variants
- `callbacks` : some available callback functions to use with implemented variants to gather convergence data
- `matrices` : test matrices