The Riemannian conjugate gradient method is a method to calculate ground states of the Kohn-Sham minimization problem, implemented using [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl). 
This is the implementation from this [preprint](https://arxiv.org/abs/2503.16225). This branch is dedicated to preserve the experiments from that paper, for the current version, refer to the `main` branch.

# Getting started

One first needs to locally save the directory `experiments` and its contents.

The Julia enviroment in the `experiments` directory is provided via the `Manifest.toml` and `Project.toml` files. It can be installed by running from the `experiment` directory
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
This also installs the `RCG_DFTK` package version from this branch.

# Using the generate_plots_models script
In order to compare the methods, run (from the `experiments` directory)
```bash
julia --project=. PPS26/generate_plots_models.jl 
```
It generates the Iterations, Hamiltonian and Times plots for all three models and also (in the `latex` case) generates the table with the percentage of runtimes caused by Hamiltonian multiplications for all methods and models. 
The user can manually set the parameters in lines 11 to 13:

* `method_names` the methods for which the experment should be run (from `["EARCG-St", "EARCG-Gr", "EARG-St", "EARG-Gr", "H1RCG", "L2RCG", "SCF"]`)
* `model_names` what models for which the experment should be run (from `["silicon", "GaAs", "TiO2"]`)
* `save_mode` how the results should be saved (either `"png"` or `"latex"`)

The figures are then saved to the `experiments` directory. For quick experimentations, it is recommended to not run `"TiO2"`, since it usually takes a quite long time.

# Using the generate_plots_gaps script

In order to run the small gaps test, run (from the `experiments` directory)
```bash
julia --project=. PPS26/generate_plots_gaps.jl 
```
It generates the gaps plot and the two performance plots of the methods for the given gaps. In lines 14 to 17 the user can manually set the following parameters:
* `as` a collection of values for the lattice constant `a`, e.g. `as = 10:0.1:11.4`
* `N_examples` how many runs per method should be performed for each value of `a`
* `model_names` what models for which the experment should be run (from `["silicon", "GaAs", "TiO2"]`)
* `save_mode` how the results should be saved (either `"png"` or `"latex"`)

The figures are then saved to the `experiments` directory. For quick experimentations, reduce the number of `as` and set `n_examples` to `1` or `2`.

## Compiling the .tex files
In order for the files to compile, one needs to define the custom colors used in the paper and the plotwidth and height

```
\definecolor{unia-purple}{RGB}{173, 0, 124}
\definecolor{light-gray}{rgb}{0.8889,0.8889,0.8889}
\definecolor{cpl1}{rgb}{0.8889,0.4356,0.2781}
\definecolor{cpl2}{rgb}{0.0,0.6056,0.9787}
\definecolor{cpl3}{rgb}{0.2422,0.6433,0.3044}
\definecolor{cpl4}{rgb}{0.2, 0.2, 0.2}

\newlength{\plotwidth}
\newlength{\plotheight}
\setlength{\plotwidth}{55mm}
\setlength{\plotheight}{50mm}
```
The tex files are (with some minor changes) the same plots that were used in the paper, when run with standard parameters, however the random initial values may differ.
