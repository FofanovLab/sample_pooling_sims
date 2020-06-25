# Sample pooling simulations
Simulations for sample pooling for DNA Sudoku, 2D Pooling, S-Stage, binary splitting by halving, and generalized binary splitting.
The simulation results provide the number of tests, steps, and pipettings required to identify all positive samples (randomly arranged) in N samples.

# Installation
Create a conda environment
```
conda create --name <myenv> --file requirements.txt
```
Activate environment
```
conda activate <myenv>
```

# Run
The simulations are available as python scripts in the `workflow/scripts` directory but a Snakemake file is available to help automate simulation runs.
```
cd workflow
snakemake --cores <N>
```
The parameters for the number of samples, the number of positive samples, reps, etc. can be modified in the snakefile or in the config file.
The config file is not automatically loaded in the snakefile so it must be passed as an argument.
```
snakemake --cores <N> --configfile ../config/config.yaml
```
