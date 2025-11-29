# Goals for week 05

1. Practice different preprocessing techniques.
2. Practice using `sklearn` pipelines.
3. Practice using `SVM`s.
4. Practice implementing a `Kernel SVM`.

## Data Science

Learning how to model data effectively.

### Task 01

**Description:**

A music company wants to increase profits by making more catchy songs. They contact your team as they need support in identifying what makes a song popular. They provide you a dataset - `music_dirty.txt` (present in the `DATA` folder in the GitHub repository).

Perform exploratory data analysis to see how the features can be used to predict a song's `popularity`.

**Acceptance criteria:**

1. An Excel file, titled `data_audit`, with **multiple sheets** is produced.

### Task 02

**Description:**

Create a regression model that predicts a song's `popularity`. Create the `model_report` file as described in `notes.md`. Conduct experiments:

- with all models we've learned so far;
- using all and some of the features;
- that show the added value of the feature `genre`.

Perform hyperparameter tuning using cross-validation.

For the best model, attach the plot showing how the residuals vary around $0$. Interpret the results.

**Acceptance criteria:**

1. An Excel file, titled `model_report`, is created which follows **all** guidelines.
2. At least two metrics are analyzed.
3. Hyperparameter tuning is performed.
4. Cross-validation is performed.

### Task 03

**Description:**

The music company provides you with another dataset - `music_dirty_missing_vals.txt`. They say that it holds more recent information but some of the values are missing.

They want to use this data for two classification tasks:

- predicting the `genre` of a song;
- predicting whether the `genre` of a song is `Rock`.

Perform exploratory data analysis to see how the features can be used to solve these two tasks. Create one plot for univariate analysis, but two plots for the multivariate one as the target values will be different. In the univariate analysis table add two columns with title `Comments` - use the first for the first prediction task and the second one - for the binary classification task.

**Acceptance criteria:**

1. An Excel file, titled `data_audit`, with **multiple sheets** is produced.

### Task 04

**Description:**

Create a classification model that predicts a whether a song is a `Rock` song or not. Create the `model_report` file as described in `notes.md`. Conduct experiments:

- with all models we've learned so far;
- using all and some of the features;
- that compare different techniques for dealing with missing values;
- that compare scaling the data vs leaving it as it is.

Show how `sklearn` pipelines can be used to create a data flow pipeline.

Perform hyperparameter tuning using cross-validation.

For the best model attach the ROC curve and the full classification report. For all models attach the confusion matrix in a cell in the corresponding row. Interpret the results.

**Acceptance criteria:**

1. An Excel file, titled `model_report`, is created which follows **all** guidelines.
2. At least two metrics are analyzed.
3. Hyperparameter tuning is performed.
4. Cross-validation is performed.
5. `sklearn` pipelines are used.

### Task 05

**Description:**

Create a classification model that predicts the `genre` of a song. Create the `model_report` file as described in `notes.md`. Conduct experiments:

- with all models we've learned so far;
- using all and some of the features;
- that compare different techniques for dealing with missing values;
- that compare scaling the data vs leaving it as it is.

Show how `sklearn` pipelines can be used to create a data flow pipeline.

Perform hyperparameter tuning using cross-validation.

For the best model attach the full classification report. For all models attach the confusion matrix in a cell in the corresponding row. Interpret the results.

**Acceptance criteria:**

1. An Excel file, titled `model_report`, is created which follows **all** guidelines.
2. At least two metrics are analyzed.
3. Hyperparameter tuning is performed.
4. Cross-validation is performed.
5. `sklearn` pipelines are used.

### Task 06

**Description:**

After communicating with the music company, they agree to do a resupply with clean data only. They provide the file `music_clean.csv`.

They want to use this data for the following tasks:

- predicting a song's `loudness`;
- predicting a song's `energy`;
- predicting whether a song will be more popular than average. This is a binary classification task - think about how you're going to encode the target variable.

Perform exploratory data analysis to see how the features can be used to solve these tasks.

**Acceptance criteria:**

1. An Excel file, titled `data_audit`, with **multiple sheets** is produced.

### Task 07

**Description:**

Create a regression model that predicts a song's `loudness`. Create the `model_report` file as described in `notes.md`. Conducts experiments that demonstrate different processing techniques, including standardization and normalization.

Show how `sklearn` pipelines can be used to create a data flow pipeline.

Perform hyperparameter tuning using cross-validation.

For the best model, attach the plot showing how the residuals vary around $0$. Interpret the results.

**Acceptance criteria:**

1. An Excel file, titled `model_report`, is created which follows **all** guidelines.
2. At least two metrics are analyzed.
3. Hyperparameter tuning is performed.
4. Cross-validation is performed.
5. `sklearn` pipelines are used.

### Task 08

**Description:**

Create a regression model that predicts a song's `energy`. Create the `model_report` file as described in `notes.md`. Conducts experiments that demonstrate different processing techniques, including standardization and normalization.

Show how `sklearn` pipelines can be used to create a data flow pipeline.

Perform hyperparameter tuning using cross-validation.

For the best model, attach the plot showing how the residuals vary around $0$. Interpret the results.

**Acceptance criteria:**

1. An Excel file, titled `model_report`, is created which follows **all** guidelines.
2. At least two metrics are analyzed.
3. Hyperparameter tuning is performed.
4. Cross-validation is performed.
5. `sklearn` pipelines are used.

### Task 09

**Description:**

Create a classification model that predicts whether a song will be more popular than average. Create the `model_report` file as described in `notes.md`. Conducts experiments that demonstrate different processing techniques, including standardization and normalization.

Show how `sklearn` pipelines can be used to create a data flow pipeline.

Perform hyperparameter tuning using cross-validation.

For the best model attach the ROC curve and the full classification report. For all models attach the confusion matrix in a cell in the corresponding row. Interpret the results.

**Acceptance criteria:**

1. An Excel file, titled `model_report`, is created which follows **all** guidelines.
2. At least two metrics are analyzed.
3. Hyperparameter tuning is performed.
4. Cross-validation is performed.
5. `sklearn` pipelines are used.

## Engineering

Building systems and implementing models.

### Task 01

**Description:**

Let's implement some kernels. Create a module `kernels.py` and in it implement the following kernels: `linear`, `polynomial`, `rbf`, `sigmoid`.

Adding unit tests is encouraged, but not strictly necessary.

**Acceptance criteria:**

1. A function `linear` is added to `ml_lib.kernels`.
2. A function `polynomial` is added to `ml_lib.kernels`.
3. A function `rbf` is added to `ml_lib.kernels`.
4. A function `sigmoid` is added to `ml_lib.kernels`.

### Task 02

**Description:**

Let's implement the `Kernel SVM` in a class `SVC`. It should have the following API:

- a method `__init__` via which the hyperparameters `C` (a float), `kernel` (a string identifier), and `gamma` (a float) can be specified;
- a method `fit` that trains the `Kernel SVM` using the package `quadprog` (if you haven't already, run `pip install -Ur requirements.txt` in the root of our repository (make sure to run `git pull` first)):
  - it should save the following values as a minimum in the state of the object: `support_vectors_` and `intercept_` (feel free to add others).
- a method `predict` that returns class labels for a matrix with observations.

**Acceptance criteria:**

1. A class `SVC` is added in the module `ml_lib.svm`.
2. The class `SVC` implements a `Kernel SVM` that can utilize the *Kernel trick*.

### Task 03

**Description:**

Use your model to recreate one of your experiments in task 9.

**Acceptance criteria:**

1. It is shown that the implementation can be used to recreate the results obtained via `sklearn`'s `SVC`.
