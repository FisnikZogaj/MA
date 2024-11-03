In ParametricGraphModels, the labelled ADC-SBM generator is stored in the ADC_SBM.py file.
Main.py calls train.py in parallel.
For each training, specified in train.py, results are stored in the ExperimentLogs folder.
Config.py holds the parameters, specifying the 6 graph scenarios.
GNN_models.py holds the 4 network architectures.
requirements.py makes sure the libraries are in the right versions (not used in the simulations, just for others to assure).
eval.ipynb is used for the final results and plots.
Final100 is the folder where all results are stored as .pkl files. They can be conveniently loaded, instead of re-running
the entire experiment.
Re-running the entire experiment, without changing the script, would still lead to the same results, due to random seeds.
In Figures.ipynb some plost are stored, used in the thesis.

