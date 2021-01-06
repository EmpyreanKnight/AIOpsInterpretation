# AIOpsInterpretation
This repository contains files for the AIOps interpretation project.

## Introduction
We organize the repository as follows:
1. Data files: due to the enormous size, I have zipped and uploaded the two CSV files in the release page of this repository.
2. Experiment code: the experiment codes are put right in this repository. 

## Guide to execute experiment code
1. Get the environment ready: Our experiment is carried out using the following packages and versions:
- Python: 3.8.3
- Numpy: 1.18.5
- Scipy: 1.5.0
- Pandas: 1.0.5
- Sklearn: 0.0
- Mkl: 2019.0
- Rgf-Python: 3.9.0
- Xgboost: 1.2.1

We recommend use an [Anaconda](https://docs.anaconda.com/anaconda/install/) enviorment with Python version 3.8.3, install the `xgboost` and `rgf-python` packages through `pip`,then everything would be ready.
If you insist to install a Python distribution from elsewhere and failed to import the `mkl` package, you could delete the two related lines in the `utilities.py` file. However, doing so would make the NN model to use unlimited CPU cores.

2. Have the data file ready: Please download and place the data files (`google_job_failure.csv` and `disk_failure_v2.csv`) and place them in the same folder with source code files. You could also have the data files in other folders with other names, but make sure to change the related variables in the `utilities.py` file.

3. Take care of configuration variables: The code contains a few variables to control the input and output folder and file names. The variables related to the data file are located at the head of the `utilities.py` file, and variables related to the experiment output file and trained model save folder are located at the head of each experiment code file. Note that the experiment code removes the old output file before execution, so ensure you have saved the previous version. If you choose to save pickled models, please make sure the folder exists.

4. Control the behavior of code: for maximum flexibility, the experiment code accepts command-line arguments to select model, dataset, iteration rounds, and whether to pickle the models. Please read the following section for details. 

5. Execution after logging out: As some experiments could take a prolonged time to finish, we recommend using `GNU Screen` or `nohup` to execute on a server. An example of executing the feature importance experiment on the `Google` data set and `RF` model in `100` iteration `without` saving trained models with `nohup` in the `background` and dump the command line output to `output.out` would be: `nohup python -u predict_feature_importance.py -d g -m rf -n 100 --save > output.out 2>&1 &`.


### Command line arguments
In our experiment code, we have the following command-line argument:
1. `-d` is a **required** parameter for choosing the dataset. Two choices are available: `g` for the Google dataset, `b` for the Backblaze dataset.
2. `-m` is a **required** parameter for choosing the model. Eight choices are available: `lda`, `qda`, `lr`, `cart`, `gbdt`, `nn`, `rf`, and `rgf`. Please note that the argument should be all *lowercase* letters.
3. `-n` is an optional parameter for the repetition time of the experiments. The default value is 100 runs, which is also the same iteration we used in our paper.
4. `--save` or `--no-save` controls whether to pickle the trained models and make them persist in the local hard drive. By default, the experiment code *saves* the models.

### Experiment code files
We have the following code files in this repository:
1. `utilities.py` contains all kinds of helper functions used by other files. Please make sure it is in the same folder of experiment code.
2. `predict_feature_importance.py` calculate the feature importance in each time period and save the results in an output CSV file. This file could embody the general workflow of our experiment.
3. `concept_drift_detection/` folder contains the code for updating (retraining use a sliding window) when detected concept drift, and an example execution file (used in another project, so some design would be different). It contains experiments for retraining based on three concept drift detection methods, an always-retrain approach, and a stationary model that never updates.
4. `ensemble_methods/` folder contains the code for updating using time-based ensemble models and an example execution file (used in another project, so some design would be different). It contains experiments for two time-based ensemble models (i.e., SEA and AWE).