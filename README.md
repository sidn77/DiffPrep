# DiffPrep

This repository contains the source code for SIGMOD 2023 paper: [DiffPrep: Differentiable Data Preprocessing Pipeline Search for Learning over Tabular Data](https://arxiv.org/pdf/2308.10915.pdf)

## Installation
Run the following command to setup a [conda](https://www.anaconda.com) environment for DiffPrep and install the required packages.

```
conda env create -f environment.yml
conda activate diffprep
```

## Run experiments
The following command will preprocess a dataset with a preprocessing method, then train and evaluate an ML model on the preprocessed data.

```
python main.py --dataset <dataset_name> --method <method_name> --model <model_name> --result_dir <result_dir>
```
### Parameters
**dataset_name**: The available dataset name can be found in the `data` folder, where each folder corresponds to one dataset. The command will run all datasets in the folder if this is not specified.

**method_name**: There are 4 available preprocessing methods.

- `default`: This approach uses a default pipeline that first imputes numerical missing values with the mean value of the column and categorical missing values with the most frequent value of the column. Then, it normalizes each feature using standardization. 
- `random`: This approach searches for a pipeline by training ML models with randomly sampled pipelines 20 times and selecting the one with the best validation accuracy.
- `diffprep_fix`: This is our approach with a pre-defined fixed transformation order.
- `diffprep_flex`: This is our approach with a flexible transformation order.

**model_name**: There are 2 available ML models.
- `log`: Logistic Regression (supported by branch `main` and `sid-1`)
- `reg`: two-layer NN (supported by branch `sid`)

**task**
- `regression`: Runs regression using MSE as the loss function
- `classification`: Runs classification with cross-entropy loss as the loss function

**result_dir**: The directory where the results will be saved. Default: `result`.

The following command will train and evaluate an ML model for a given dataset and pipeline configuration defined by alpha and beta matrices. This experiment requires the checking out of branch `run_pipe` before execution.

```
python main_baseline.py --dataset <dataset_name> --model <model_name> --method fixed
```

### Parameters
**dataset_name**: The available dataset name can be found in the `data` folder, where each folder corresponds to one dataset. The command will run all datasets in the folder if this is not specified.

**method_name**: There are 4 available preprocessing methods.

- `default`: This approach uses a default pipeline that first imputes numerical missing values with the mean value of the column and categorical missing values with the most frequent value of the column. Then, it normalizes each feature using standardization. 
- `random`: This approach searches for a pipeline by training ML models with randomly sampled pipelines 20 times and selecting the one with the best validation accuracy.
- `diffprep_fix`: This is our approach with a pre-defined fixed transformation order.
- `diffprep_flex`: This is our approach with a flexible transformation order.
- `fixed`: This is the approach with a user-defined (using the alpha and beta matrices) transformation order and operation type.

**model_name**: There are 2 available ML models.
- `log`: Logistic Regression
- `reg`: two-layer NN

**result_dir**: The directory where the results will be saved. Default: `result`.

### Experiment Setup
**Hyperparameter Tuning**: The hyperparameters of the experiment are specified in the `main.py`. By default, we tune the learning rate in our experiments using the hold-out validation set. 

**Early Stopping**: We adopt earling stopping in our training process, where we keep track of the validation accuracy in our experiments and terminate training when the validation accuracy cannot be improved significantly.

### Results
The results will be saved in the folder `<result_dir>/<method_name>/<dataset_name>`. There are two resulting files:

- `result.json`:  This file stores the test accuracy of the best epoch selected by the validation loss.

- `params.json`: This file stores the model hyperparameters.

- `bestpipelines.json`: This file stores the alpha and beta matrices after converting the logits to probabilities corresponding to the model with highest test accuracy.   


# Visualizations and Distance Calculation
For running the notebooks the branch `run_pipe` will need to be checked out. For each dataset, two bestpipelines.json files will be generated. For example, for the abalone dataset, the following two files will be created:

bestpipelines_aba_c.json – corresponding to classification
bestpipelines_aba_r.json – corresponding to regression

For each visualization file, (`pca.ipynb`, `mds.ipynb`, `t_sne.ipynb`) update the following:

file_path_pc = bestpipelines_aba_c.json  - path to classification file 
file_path_pr = bestpipelines_aba_r.json  - path to regression file 

Then run the visualization file that was just updated. This will generate the 2-d visualization and print out the distance values for the 5 distance metrics, and the total number of features. For example,

bestpipelines_mozilla_c.json bestpipelines_mozilla_r.json 
[cosine_totaldist, euclid_totaldist, manhattan_totaldist, chebyshev_totaldist, canberra_totaldist] 
[11.120676452028675, 23.567524488152976, 29.675894607101863, 21.936217949550354, 16.971087403210998].

To calculate the Pearson correlation coefficient:

Divide the distances obtained by the number of features. The number of features will be printed out when you run the `tsne.ipynb` visualization file.

Run 
```
from scipy.stats import pearsonr 
scipy.stats.pearsonr(x, y)
```

where `x` is the the drop in accuracy across the different datasets and `y`  is the average distance values across datasets for a given visualization and distance metric combination.

## Link to the Kaggle notebooks that were used to run the experiments using Diffprep
https://www.kaggle.com/sidn77/code


