# Table of Contents

- [Table of Contents](#table-of-contents)
- [Week 01 - Numpy, Pandas, Matplotlib \& Seaborn](#week-01---numpy-pandas-matplotlib--seaborn)
  - [Administrative](#administrative)
  - [Python Packages](#python-packages)
    - [Introduction](#introduction)
    - [Importing means executing the script](#importing-means-executing-the-script)
  - [Code formatting](#code-formatting)
  - [NumPy](#numpy)
    - [0. Importing](#0-importing)
    - [1. Array Creation](#1-array-creation)
    - [2. Array Attributes and Inspection](#2-array-attributes-and-inspection)
    - [3. Array Manipulation](#3-array-manipulation)
    - [4. Indexing, Slicing, and Iteration](#4-indexing-slicing-and-iteration)
    - [5. Mathematical and Universal Functions (`ufuncs`)](#5-mathematical-and-universal-functions-ufuncs)
    - [6. Linear Algebra](#6-linear-algebra)
    - [7. Statistical Functions](#7-statistical-functions)
    - [8. Broadcasting](#8-broadcasting)
    - [9. Random Number Generation](#9-random-number-generation)
    - [10. File I/O and Miscellaneous](#10-file-io-and-miscellaneous)
  - [Pandas](#pandas)
    - [0. Importing](#0-importing-1)
    - [1. Introduction to DataFrames](#1-introduction-to-dataframes)
    - [2. Data Structures and Creation](#2-data-structures-and-creation)
    - [3. Data Inspection](#3-data-inspection)
    - [4. Data Selection and Indexing](#4-data-selection-and-indexing)
    - [5. Filtering Data](#5-filtering-data)
    - [6. Data Manipulation](#6-data-manipulation)
    - [7. Statistical and Mathematical Operations](#7-statistical-and-mathematical-operations)
    - [8. Handling Missing Data](#8-handling-missing-data)
    - [9. Grouping and Aggregation](#9-grouping-and-aggregation)
    - [10. File I/O and Miscellaneous](#10-file-io-and-miscellaneous-1)
  - [Matplotlib](#matplotlib)
    - [Line plot](#line-plot)
    - [Scatter plot](#scatter-plot)
    - [Drawing multiple plots on one figure](#drawing-multiple-plots-on-one-figure)
    - [The logarithmic scale](#the-logarithmic-scale)
    - [Histogram](#histogram)
      - [Introduction](#introduction-1)
      - [In `matplotlib`](#in-matplotlib)
      - [Use cases](#use-cases)
    - [Checkpoint](#checkpoint)
    - [Customization](#customization)
      - [Axis labels](#axis-labels)
      - [Title](#title)
      - [Ticks](#ticks)
      - [Adding more data](#adding-more-data)
      - [`plt.tight_layout()`](#plttight_layout)
        - [Problem](#problem)
        - [Solution](#solution)
  - [Seaborn](#seaborn)
    - [0. Importing](#0-importing-2)
    - [1. Figure, Axes, and Styling](#1-figure-axes-and-styling)
    - [2. Basic Plots: Line and Scatter](#2-basic-plots-line-and-scatter)
    - [3. Categorical Plots: Bar and Box](#3-categorical-plots-bar-and-box)
    - [4. Distribution Plots: Histogram and KDE](#4-distribution-plots-histogram-and-kde)
    - [5. Advanced Statistical Plots](#5-advanced-statistical-plots)
      - [Pair Plot](#pair-plot)
      - [Heatmap](#heatmap)
      - [Violin Plot](#violin-plot)
    - [6. Customization and Subplots](#6-customization-and-subplots)
    - [7. Saving and Displaying Plots](#7-saving-and-displaying-plots)
- [Week 02 - Machine learning with scikit-learn](#week-02---machine-learning-with-scikit-learn)
  - [What is machine learning?](#what-is-machine-learning)
  - [The `scikit-learn` syntax](#the-scikit-learn-syntax)
  - [The classification challenge](#the-classification-challenge)
  - [Measuring model performance](#measuring-model-performance)
  - [Model complexity (overfitting and underfitting)](#model-complexity-overfitting-and-underfitting)
  - [Hyperparameter optimization (tuning) / Model complexity curve](#hyperparameter-optimization-tuning--model-complexity-curve)
  - [The Model Report](#the-model-report)
- [Week 03 - Regression](#week-03---regression)
  - [Regression problems](#regression-problems)
  - [Regression mechanics](#regression-mechanics)
  - [Modelling via regression](#modelling-via-regression)
    - [Data Preparation](#data-preparation)
    - [Modelling](#modelling)
  - [Model evaluation](#model-evaluation)
    - [Visually](#visually)
    - [Using a metric](#using-a-metric)
      - [Adjusted $R^2$ ($R\_{adj}^2$)](#adjusted-r2-r_adj2)
    - [Using a loss function](#using-a-loss-function)
- [Week 04 - Regularized Regression, Logistic Regression, Cross Validation](#week-04---regularized-regression-logistic-regression-cross-validation)
  - [Logistic Regression](#logistic-regression)
    - [Binary classification](#binary-classification)
    - [Model fitting](#model-fitting)
    - [Multiclass classification](#multiclass-classification)
    - [Logistic regression in `scikit-learn`](#logistic-regression-in-scikit-learn)
  - [Regularized Regression](#regularized-regression)
    - [Regularization](#regularization)
    - [Ridge Regression](#ridge-regression)
    - [Lasso Regression](#lasso-regression)
    - [Feature Importance](#feature-importance)
    - [Lasso Regression and Feature Importance](#lasso-regression-and-feature-importance)
  - [Classification Metrics](#classification-metrics)
    - [A problem with using `accuracy` always](#a-problem-with-using-accuracy-always)
    - [Confusion matrix in scikit-learn](#confusion-matrix-in-scikit-learn)
  - [The receiver operating characteristic curve (`ROC` curve)](#the-receiver-operating-characteristic-curve-roc-curve)
    - [In `scikit-learn`](#in-scikit-learn)
    - [The Area Under the Curve (`AUC`)](#the-area-under-the-curve-auc)
    - [In `scikit-learn`](#in-scikit-learn-1)
  - [Cross Validation](#cross-validation)
  - [Hyperparameter optimization/tuning](#hyperparameter-optimizationtuning)
    - [Hyperparameters](#hyperparameters)
    - [Introduction](#introduction-2)
    - [Grid search cross-validation](#grid-search-cross-validation)
    - [In `scikit-learn`](#in-scikit-learn-2)
    - [Randomized search cross-validation](#randomized-search-cross-validation)
      - [Benefits](#benefits)
    - [Evaluating on the test set](#evaluating-on-the-test-set)
- [Week 05 - Preprocessing and Pipelines. Support Vector Machines](#week-05---preprocessing-and-pipelines-support-vector-machines)
  - [Preprocessing and Pipelines](#preprocessing-and-pipelines)
    - [Dealing with categorical features](#dealing-with-categorical-features)
      - [Dropping one of the categories per feature](#dropping-one-of-the-categories-per-feature)
      - [In `scikit-learn` and `pandas`](#in-scikit-learn-and-pandas)
    - [EDA with categorical feature](#eda-with-categorical-feature)
    - [Handling missing data](#handling-missing-data)
      - [Removing missing values](#removing-missing-values)
      - [Imputing missing values](#imputing-missing-values)
      - [Using pipelines](#using-pipelines)
    - [Centering and scaling](#centering-and-scaling)
    - [Standardization and normalization](#standardization-and-normalization)
    - [Scaling in `scikit-learn`](#scaling-in-scikit-learn)
    - [How do we decide which model to try out in the first place?](#how-do-we-decide-which-model-to-try-out-in-the-first-place)
  - [Support Vector Machines](#support-vector-machines)
    - [The use-case for kernel SVMs](#the-use-case-for-kernel-svms)
    - [Prediction function](#prediction-function)
    - [Fitting the model](#fitting-the-model)
    - [The Kernel Trick](#the-kernel-trick)
    - [Minimizing the objective](#minimizing-the-objective)
    - [`SVC` in `scikit-learn`](#svc-in-scikit-learn)
    - [`LinearSVC` in `scikit-learn`](#linearsvc-in-scikit-learn)
    - [Loss function diagrams](#loss-function-diagrams)
    - [Comparing logistic regression and SVM](#comparing-logistic-regression-and-svm)
    - [`SGDClassifier`](#sgdclassifier)
- [Week 06 - The Bias-Variance Tradeoff. Decision Trees](#week-06---the-bias-variance-tradeoff-decision-trees)
  - [Decision Trees](#decision-trees)
    - [Classification-And-Regression-Tree (`CART`)](#classification-and-regression-tree-cart)
      - [Classification tree](#classification-tree)
        - [What is a classification tree?](#what-is-a-classification-tree)
        - [How does a classification tree learn?](#how-does-a-classification-tree-learn)
        - [What criterion is used to measure the impurity of a node?](#what-criterion-is-used-to-measure-the-impurity-of-a-node)
      - [Regression tree](#regression-tree)
    - [The Bias-Variance Tradeoff](#the-bias-variance-tradeoff)
      - [The goals of Supervised Learning](#the-goals-of-supervised-learning)
      - [Difficulties in Approximating $f$](#difficulties-in-approximating-f)
      - [Generalization Error](#generalization-error)
      - [Model Complexity](#model-complexity)
      - [Bias-Variance Tradeoff: A Visual Explanation](#bias-variance-tradeoff-a-visual-explanation)
      - [Checkpoint](#checkpoint-1)
    - [Train-test split revisited](#train-test-split-revisited)
      - [Estimating the Generalization Error](#estimating-the-generalization-error)
      - [Diagnose Variance Problems](#diagnose-variance-problems)
      - [Diagnose Bias Problems](#diagnose-bias-problems)
    - [Ensemble Learning](#ensemble-learning)
      - [Advantages of CARTs](#advantages-of-carts)
      - [Limitations of CARTs](#limitations-of-carts)
      - [What is Ensemble Learning?](#what-is-ensemble-learning)
      - [The `Voting Classifier`](#the-voting-classifier)
- [Week 07 - Bagging. Boosting](#week-07---bagging-boosting)
  - [Bagging](#bagging)
    - [Bootstrap](#bootstrap)
    - [Bagging: Training](#bagging-training)
    - [Bagging: Prediction](#bagging-prediction)
    - [Bagging Reduces Variance](#bagging-reduces-variance)
    - [The `oob_score_`](#the-oob_score_)
    - [Further Diversity with `Random Forests`](#further-diversity-with-random-forests)
    - [Feature Importance with Trees and Forests](#feature-importance-with-trees-and-forests)
  - [Boosting](#boosting)
    - [Adaboost](#adaboost)
    - [Learning Rate](#learning-rate)
    - [AdaBoost: Prediction](#adaboost-prediction)
    - [Gradient Boosted Trees](#gradient-boosted-trees)
    - [Gradient Boosted Trees for Regression: Training](#gradient-boosted-trees-for-regression-training)
    - [Shrinkage](#shrinkage)
    - [Gradient Boosted Trees for Regression: Prediction](#gradient-boosted-trees-for-regression-prediction)
    - [Gradient Boosting: Cons](#gradient-boosting-cons)
    - [Stochastic Gradient Boosting](#stochastic-gradient-boosting)
- [Week 08 - Unsupervised Learning](#week-08---unsupervised-learning)
  - [The added value of unsupervised learning](#the-added-value-of-unsupervised-learning)
  - [The `K-Means Clustering` algorithm](#the-k-means-clustering-algorithm)
  - [DBScan and HDBScan](#dbscan-and-hdbscan)
  - [Scaling features for better clusters](#scaling-features-for-better-clusters)
  - [Visualization communicates insight](#visualization-communicates-insight)
    - [`Agglomerative` hierarchical clustering](#agglomerative-hierarchical-clustering)
    - [t-distributed stochastic neighbor embedding (`t-SNE`)](#t-distributed-stochastic-neighbor-embedding-t-sne)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
  - [`TruncatedSVD` and `TF-IDF`](#truncatedsvd-and-tf-idf)
  - [Non-negative matrix factorization (NMF)](#non-negative-matrix-factorization-nmf)
    - [NMF features](#nmf-features)
    - [NMF components are topics](#nmf-components-are-topics)
    - [Grayscale images](#grayscale-images)
    - [Encoding a collection of images](#encoding-a-collection-of-images)
    - [Building recommender systems using NMF](#building-recommender-systems-using-nmf)
- [Week 09 - Natural Language Processing](#week-09---natural-language-processing)
  - [The added value of NLP](#the-added-value-of-nlp)
  - [Regular expressions](#regular-expressions)
  - [Tokenization](#tokenization)
  - [Bag-of-words](#bag-of-words)
  - [Text preprocessing](#text-preprocessing)
  - [Gensim](#gensim)
  - [Named Entity Recognition](#named-entity-recognition)
    - [Using `nltk`](#using-nltk)
    - [What is SpaCy?](#what-is-spacy)
  - [The Naive Bayes Classifier](#the-naive-bayes-classifier)
- [Week 10 - Hello, Deep Learning. Implementing a Multilayer Perceptron](#week-10---hello-deep-learning-implementing-a-multilayer-perceptron)
  - [So what can deep learning models model?](#so-what-can-deep-learning-models-model)
  - [Modeling a neuron that can multiply by `2`](#modeling-a-neuron-that-can-multiply-by-2)
- [Week 11 - Implementing Gradient Descent](#week-11---implementing-gradient-descent)
  - [Backpropagation](#backpropagation)
  - [Topological sort](#topological-sort)
  - [The hyperbolic tangent](#the-hyperbolic-tangent)
  - [Python OOP (Magic Methods)](#python-oop-magic-methods)
    - [Initialization and Construction](#initialization-and-construction)
    - [Arithmetic operators](#arithmetic-operators)
    - [String Magic Methods](#string-magic-methods)
    - [Comparison magic methods](#comparison-magic-methods)
- [Week 12 - Hello, PyTorch](#week-12---hello-pytorch)
  - [PyTorch Dataset](#pytorch-dataset)
  - [PyTorch Dataloader](#pytorch-dataloader)
  - [PyTorch Model](#pytorch-model)
  - [Training loop - PyTorch style](#training-loop---pytorch-style)
  - [Optimizers](#optimizers)
  - [Model evaluation - PyTorch style](#model-evaluation---pytorch-style)
    - [Unstable gradients](#unstable-gradients)
    - [Solutions to unstable gradients](#solutions-to-unstable-gradients)
      - [Proper weights initialization](#proper-weights-initialization)
      - [More appropriate activation functions](#more-appropriate-activation-functions)
      - [Batch normalization](#batch-normalization)
- [Week 13 - Convolutional Neural Networks](#week-13---convolutional-neural-networks)
  - [The Clouds dataset](#the-clouds-dataset)
  - [Converting pixels to tensors and tensors to pixels](#converting-pixels-to-tensors-and-tensors-to-pixels)
    - [`ToTensor()`](#totensor)
    - [`PILToTensor()`](#piltotensor)
    - [`ToPILImage()`](#topilimage)
  - [Loading images to PyTorch](#loading-images-to-pytorch)
  - [Data augmentation](#data-augmentation)
  - [CNNs - The neural networks for image processing](#cnns---the-neural-networks-for-image-processing)
  - [Architecture](#architecture)
  - [Cross-Entropy loss](#cross-entropy-loss)
  - [Checkpoint](#checkpoint-2)
  - [Precision \& Recall for Multiclass Classification (revisited)](#precision--recall-for-multiclass-classification-revisited)
    - [Computing total value](#computing-total-value)
    - [Computing per class value](#computing-per-class-value)
- [Week 14 - Recurrent Neural Networks](#week-14---recurrent-neural-networks)
  - [Sequential data](#sequential-data)
  - [Electricity consumption prediction](#electricity-consumption-prediction)
  - [`TensorDataset`](#tensordataset)
  - [The recurrent neuron](#the-recurrent-neuron)
      - [Internals of the `RNN` cell](#internals-of-the-rnn-cell)
      - [Unrolling through time](#unrolling-through-time)
      - [Deep RNNs](#deep-rnns)
    - [Architecture types](#architecture-types)
      - [Sequence-to-sequence architecture](#sequence-to-sequence-architecture)
      - [Sequence-to-vector architecture](#sequence-to-vector-architecture)
      - [Vector-to-sequence architecture](#vector-to-sequence-architecture)
      - [Encoder-decoder architecture](#encoder-decoder-architecture)
      - [Choosing the right architecture for the problem](#choosing-the-right-architecture-for-the-problem)
    - [The problem with using plain RNNs the model sequential data](#the-problem-with-using-plain-rnns-the-model-sequential-data)
    - [Internals of the `LSTM` cell](#internals-of-the-lstm-cell)
    - [Internals of the `GRU` cell](#internals-of-the-gru-cell)
    - [Should I use RNN, LSTM, or GRU?](#should-i-use-rnn-lstm-or-gru)
- [Week 15 - Bigram language model using counting](#week-15---bigram-language-model-using-counting)
  - [`PyCaret`: Low-code Machine Learning](#pycaret-low-code-machine-learning)
  - [Generating human names](#generating-human-names)
  - [Character-level language model](#character-level-language-model)
  - [An $n$-gram language model](#an-n-gram-language-model)
  - [Building out a character-level bigram language model](#building-out-a-character-level-bigram-language-model)
  - [The Multinomial Distribution](#the-multinomial-distribution)
  - [Sampling from a multinomial distribution with PyTorch](#sampling-from-a-multinomial-distribution-with-pytorch)
    - [The `Generator` class in PyTorch](#the-generator-class-in-pytorch)
    - [Sampling with a `Generator`](#sampling-with-a-generator)
  - [Negative Log Likelihood](#negative-log-likelihood)
  - [Model Smoothing and Regularization](#model-smoothing-and-regularization)

# Week 01 - Numpy, Pandas, Matplotlib & Seaborn

## Administrative

- [ ] Create a chat in Messenger.

## Python Packages

### Introduction

You write all of your code to one and the same Python script.

<details>

<summary>What are the problems that arise from that?</summary>

- Huge code base: messy;
- Lots of code you won't use;
- Maintenance problems.

</details>

<details>

<summary>How do we solve this problem?</summary>

We can split our code into libraries (or in the Python world - **packages**).

Packages are a directory of Python scripts.

Each such script is a so-called **module**.

Here's the hierarchy visualized:

![w01_packages_modules.png](./assets/w01_packages_modules.png "w01_packages_modules.png")

These modules specify functions, methods and new Python types aimed at solving particular problems. There are thousands of Python packages available from the Internet. Among them are packages for data science:

- there's **NumPy to efficiently work with arrays**;
- **Matplotlib for data visualization**;
- **scikit-learn for machine learning**.

</details>

Not all of them are available in Python by default, though. To use Python packages, you'll first have to install them on your own system, and then put code in your script to tell Python that you want to use these packages. Advice:

- always install packages in **virtual environments** (abstractions that hold packages for separate projects).
  - You can create a virtual environment by using the following code:

    ```console
    python3 -m venv .venv
    ```

    This will create a hidden folder, called `.venv`, that will store all packages you install for your current project (instead of installing them globally on your system).

  - If there is a `requirements.txt` file, use it to install the needed packages beforehand.
    - In the github repo, there is such a file - you can use it to install all the packages you'll need in the course. This can be done by using this command:

    ```console
    (if on Windows) > .venv\Scripts\activate
    (if on Linux) > source .venv/bin/activate
    (.venv) > pip install -r requirements.txt
    ```

Now that the package is installed, you can actually start using it in one of your Python scripts. To do this you should import the package, or a specific module of the package.

You can do this with the `import` statement. To import the entire `numpy` package, you can do `import numpy`. A commonly used function in NumPy is `array`. It takes a Python list as input and returns a [`NumPy array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) object as an output. The NumPy array is very useful to do data science, but more on that later. Calling the `array` function like this, though, will generate an error:

```python
import numpy
array([1, 2, 3])
```

```console
NameError: name `array` is not defined
```

To refer to the `array` function from the `numpy` package, you'll need this:

```python
import numpy
numpy.array([1, 2, 3])
```

```console
array([1, 2, 3])
```

This time it works.

Using this `numpy.` prefix all the time can become pretty tiring, so you can also import the package and refer to it with a different name. You can do this by extending your `import` statement with `as`:

```python
import numpy as np
np.array([1, 2, 3])
```

```console
array([1, 2, 3])
```

Now, instead of `numpy.array`, you'll have to use `np.array` to use NumPy's functions.

There are cases in which you only need one specific function of a package. Python allows you to make this explicit in your code.

Suppose that we ***only*** want to use the `array` function from the NumPy package. Instead of doing `import numpy`, you can instead do `from numpy import array`:

```python
from numpy import array
array([1, 2, 3])
```

```console
array([1, 2, 3])
```

This time, you can simply call the `array` function without `numpy.`.

This `from import` version to use specific parts of a package can be useful to limit the amount of coding, but you're also loosing some of the context. Suppose you're working in a long Python script. You import the array function from numpy at the very top, and way later, you actually use this array function. Somebody else who's reading your code might have forgotten that this array function is a specific NumPy function; it's not clear from the function call.

![w01_from_numpy.png](./assets/w01_from_numpy.png "w01_from_numpy.png")

^ using numpy, but not very clear

Thus, the more standard `import numpy as np` call is preferred: In this case, your function call is `np.array`, making it very clear that you're working with NumPy.

![w01_import_as_np.png](./assets/w01_import_as_np.png "w01_import_as_np.png")

- Suppose you want to use the function `inv()`, which is in the `linalg` subpackage of the `scipy` package. You want to be able to use this function as follows:

    ```python
    my_inv([[1,2], [3,4]])
    ```

    Which import statement will you need in order to run the above code without an error?

  - A. `import scipy`
  - B. `import scipy.linalg`
  - C. `from scipy.linalg import my_inv`
  - D. `from scipy.linalg import inv as my_inv`

    <details>

    <summary>Reveal answer:</summary>

    Answer: D

    </details>

### Importing means executing the script

Remember that importing a package is equivalent to executing it. Thus, you should always have `if __name__ == '__main__'` block of code and call your functions from there.

Run the scripts `test_script1.py` and `test_script2.py` that are in the folder `Week_01 - Numpy, Pandas, Matplotlib, Seaborn` to see the differences.

## Code formatting

In this course we'll strive to learn how to develop scripts in Python. In general, good code in software engineering is one that is:

1. Easy to read.
2. Safe from bugs.
3. Ready for change.

This section focuses on the first point - how do we make our code easier to read? Here are some principles:

1. Use a linter/formatter.
2. Simple functions - every function should do one thing. This is the single responsibility principle.
3. Break up complex logic into multiple steps. In other words, prefer shorter lines instead of longer.
4. Do not do extended nesting. Instead of writing nested `if` clauses, prefer [`match`](https://docs.python.org/3/tutorial/controlflow.html#match-statements) or many `if` clauses on a single level.

You can automatically handle the first point - let's see how to install and use the `yapf` formatter extension in VS Code.

1. Open the `Extensions` tab, either by using the UI or by pressing `Ctrl + Shift + x`. You'll see somthing along the lines of:
  
![w01_yapf_on_vscode.png](./assets/w01_yapf_on_vscode.png "w01_yapf_on_vscode.png")

2. Search for `yapf`:

![w01_yapf_on_vscode_1.png](./assets/w01_yapf_on_vscode_1.png "w01_yapf_on_vscode_1.png")

3. Select and install it:

![w01_yapf_on_vscode_2.png](./assets/w01_yapf_on_vscode_2.png "w01_yapf_on_vscode_2.png")

4. After installing, please apply it on every Python file. To do so, press `F1` and type `Format Document`. The script would then be formatted accordingly.

![w01_yapf_on_vscode_3.png](./assets/w01_yapf_on_vscode_3.png "w01_yapf_on_vscode_3.png")

## NumPy

NumPy is a fundamental Python library for numerical computing, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.

### 0. Importing

By convention `numpy` is imported with the alias `np`:

```python
import numpy as np
```

### 1. Array Creation

These functions create new arrays from scratch or from existing data.

Use the official documentation of NumPy: <https://numpy.org/doc/stable/reference/arrays.ndarray.html#the-n-dimensional-array-ndarray>, to fill in the missing parts (marked with `???`) below.

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `np.array(object, dtype=None)` | Converts lists, tuples, or other iterables into NumPy arrays. | `np.array([[1, 2], [3, 4]])`  # Creates a 2x2 array |
| `np.zeros(shape, dtype=float)` | Creates arrays filled with zeros. | `???`  # 2x3 array of zeros |
| `np.ones(shape, dtype=None)` | ??? | `???`  # 3x2 array of integer ones |
| `np.full(shape, fill_value, dtype=None)` | ??? | `np.full((2, 2), 5)`  # ??? |
| `np.eye(N, M=None, k=0, dtype=float)` | ??? | `???`  # a matrix with 1 on the upper-right diagonal |
| `np.identity(n, dtype=None)` | ??? | `np.identity(4)`  # ??? |
| `np.arange(start=0, stop, step=1, dtype=None)` | Generates evenly spaced values within a range, like Python's `range`. | `???`  # array([0, 2, 4, 6, 8]) |
| `np.linspace(start, stop, num=50, endpoint=True)` | Creates evenly spaced samples over an interval. | `???`  # array([0., 0.25, 0.5, 0.75, 1.]) |
| `np.logspace(start, stop, num=50, base=10.0)` | Generates logarithmically spaced numbers, for exponential scales. | `np.logspace(0, 2, 3)`  # ??? |

<details>
<summary>Reveal answer</summary>

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `np.array(object, dtype=None)` | Converts lists, tuples, or other iterables into NumPy arrays. | `np.array([[1, 2], [3, 4]])`  # Creates a 2x2 array |
| `np.zeros(shape, dtype=float)` | Creates arrays filled with zeros. | `np.zeros((2, 3))`  # 2x3 array of zeros |
| `np.ones(shape, dtype=None)` | Creates arrays filled with ones. | `np.ones((3, 2), dtype=int)`  # 3x2 array of integer ones |
| `np.full(shape, fill_value, dtype=None)` | Fills arrays with a specified value. | `np.full((2, 2), 5)`  # 2x2 array filled with 5 |
| `np.eye(N, M=None, k=0, dtype=float)` | Return a 2-D array with ones on a diagonal and zeros elsewhere. | `np.eye(3, k=1)`  # a matrix with 1 on the upper-right diagonal |
| `np.identity(n, dtype=None)` | Similar to `eye` but always square with `1` on the main diagonal. | `np.identity(4)`  # 4x4 identity matrix |
| `np.arange(start=0, stop, step=1, dtype=None)` | Generates evenly spaced values within a range, like Python's `range`. | `np.arange(0, 10, 2)`  # array([0, 2, 4, 6, 8]) |
| `np.linspace(start, stop, num=50, endpoint=True)` | Creates evenly spaced samples over an interval. | `np.linspace(0, 1, 5)`  # array([0., 0.25, 0.5, 0.75, 1.]) |
| `np.logspace(start, stop, num=50, base=10.0)` | Generates logarithmically spaced numbers, for exponential scales. | `np.logspace(0, 2, 3)`  # array([1., 10., 100.]) |

</details>

### 2. Array Attributes and Inspection

These attributes provide metadata about arrays.

Use the official documentation of NumPy: <https://numpy.org/doc/stable/reference/arrays.ndarray.html#the-n-dimensional-array-ndarray>, to fill in the missing parts (marked with `???`) below.

```python
arr = np.array([[1, 2, 3]])
```

| Attribute | Problem Solved | Example |
|--------------------|----------------|---------|
| `.shape` | ??? | `arr.shape`  # ??? |
| `.ndim` | ??? | `arr.ndim`  # ??? |
| `.size` | ??? | `arr.size`  # ??? |
| `.dtype` | ??? | `arr.dtype`  # ??? |

<details>
<summary>Reveal answer</summary>

| Attribute | Problem Solved | Example |
|--------------------|----------------|---------|
| `.shape` | Returns the dimensions of the array. | `arr.shape`  # (1, 3) |
| `.ndim` | Gives the number of dimensions. | `arr.ndim`  # 2 (since the array is a matrix: 1 row, 3 columns) |
| `.size` | Total number of elements. | `arr.size`  # 3 |
| `.dtype` | Data type of elements. | `arr.dtype`  # dtype('int64') |

</details>

### 3. Array Manipulation

These functions reshape, combine, or split arrays, addressing data restructuring needs.

Use the official documentation of NumPy: <https://numpy.org/doc/stable/reference/arrays.ndarray.html#the-n-dimensional-array-ndarray>, to fill in the missing parts (marked with `???`) below.

```python
arr = np.arange(6)
```

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `arr.reshape(newshape)` | Changes array shape **without copying** data. | `reshaped = arr.reshape(2, 3); reshaped`  # ??? |
| `arr.ravel(order='C')` | ??? | `reshaped.ravel()`  # ??? |
| `arr.flatten(order='C')` | ??? | `reshaped.flatten()`  # array([0,1,2,3,4,5]) |
| `arr.T` or `arr.transpose(*axes)` | ??? | `reshaped.T`  # ??? |
| `np.concatenate((a1, a2, ...), axis=0)` | ??? | `np.concatenate((np.ones(2), np.zeros(2)))`  # ??? |
| `np.vstack(tup)` | ??? | `???`  # 2x3 array |
| `np.hstack(tup)` | ??? | `???`  # 2x2 array |
| `np.split(ary, indices_or_sections, axis=0)` | ??? | `???`  # [array([0,1]), array([2,3]), array([4,5])] |
| `np.repeat(a, repeats, axis=None)` | ??? | `np.repeat([1,2], 2)`  # array([1,1,2,2]) |
| `np.pad(array, pad_width, mode='constant')` | ??? | `np.pad(np.ones((2,2)), 1)`  # ??? |
| `np.diag(v, k=0)` | ??? | `np.diag(np.array([[1,2],[3,4]]))`  # ??? <br> `np.diag([1,2,3])`  # ??? |

<details>
<summary>Reveal answer</summary>

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `arr.reshape(newshape)` | Changes array shape **without copying** data. | `reshaped = arr.reshape(2, 3); reshaped`  # `[[0,1,2],[3,4,5]]` |
| `arr.ravel(order='C')` | Flattens multi-dimensional array into 1D. A copy is made only if needed. | `reshaped.ravel()`  # array([0,1,2,3,4,5]) |
| `arr.flatten(order='C')` | Similar to `ravel` but always copies. | `reshaped.flatten()`  # array([0,1,2,3,4,5]) |
| `arr.T` or `arr.transpose(*axes)` | Swaps axes, essential for matrix operations like dot products. | `reshaped.T`  # `[[0,3],[1,4],[2,5]]` |
| `np.concatenate((a1, a2, ...), axis=0)` | Joins arrays along an axis, for merging datasets. | `np.concatenate((np.ones(2), np.zeros(2)))`  # array([1.,1.,0.,0.]) |
| `np.vstack(tup)` | Stacks arrays vertically (row-wise), for building matrices from rows. | `np.vstack((np.arange(3), np.arange(3)))`  # 2x3 array |
| `np.hstack(tup)` | Stacks horizontally (column-wise), for adding columns. | `np.hstack((np.ones((2,1)), np.zeros((2,1))))`  # 2x2 array |
| `np.split(ary, indices_or_sections, axis=0)` | Splits array into sub-arrays, for partitioning data. | `np.split(np.arange(6), 3)`  # [array([0,1]), array([2,3]), array([4,5])] |
| `np.repeat(a, repeats, axis=None)` | Repeats elements for upsampling. | `np.repeat([1,2], 2)`  # array([1,1,2,2]) |
| `np.pad(array, pad_width, mode='constant')` | Adds padding to arrays. | `np.pad(np.ones((2,2)), 1)`  # 4x4 with zeros around |
| `np.diag(v, k=0)` | Extracts diagonal or constructs a diagonal matrix. | `np.diag(np.array([[1,2],[3,4]]))`  # array([1,4]) <br> `np.diag([1,2,3])`  # 3x3 matrix with [1,2,3] on the main diagonal |

</details>

### 4. Indexing, Slicing, and Iteration

These allow accessing and modifying subsets, solving efficient data extraction problems.

Use the official documentation of NumPy: <https://numpy.org/doc/stable/reference/arrays.ndarray.html#the-n-dimensional-array-ndarray>, to fill in the missing parts (marked with `???`) below.

```python
arr = np.arange(10)
```

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `arr[start:stop:step]` | Extracts sub-arrays, like list slicing but multi-dimensional. | `arr[2:7:2]`  # ??? |
| `arr[[indices]]` | Selects specific elements by index lists, for non-contiguous access. | `???`  # array([0,3,5]) |
| `arr[condition]` | Filters based on conditions. | `arr[arr > 5]`  # ??? <br> `???` # array([6, 8])|
| `np.where(condition, x, y)` | ??? | `np.where(arr > 5, arr, 0)`  # ??? |
| `np.argwhere(a)` | ??? | `np.argwhere(arr > 5)`  # ??? |

<details>
<summary>Reveal answer</summary>

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `arr[start:stop:step]` | Extracts sub-arrays, like list slicing but multi-dimensional. | `arr[2:7:2]`  # array([2,4,6]) |
| `arr[[indices]]` | Selects specific elements by index lists, for non-contiguous access. | `arr[[0,3,5]]`  # array([0,3,5]) |
| `arr[condition]` | Filters based on conditions. | `arr[arr > 5]`  # array([6,7,8,9]) <br> `arr[(arr > 5) & (arr % 2 == 0)]` # array([6, 8])|
| `np.where(condition, x, y)` | Conditional element-wise selection, like ternary operators for arrays. | `np.where(arr > 5, arr, 0)`  # array([0, 0, 0, 0, 0, 0, 6, 7, 8, 9]) |
| `np.argwhere(a)` | Finds indices of non-zero elements, for locating matches. | `np.argwhere(arr > 5)`  # `array([[6],[7],[8],[9]])` |

</details>

### 5. Mathematical and Universal Functions (`ufuncs`)

Element-wise operations on arrays, solving **vectorized computations** for speed over loops.

Use the official documentation of NumPy: <https://numpy.org/doc/stable/reference/arrays.ndarray.html#the-n-dimensional-array-ndarray>, to fill in the missing parts (marked with `???`) below.

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `x1 + x2` or `np.add(x1, x2)` | ??? | `np.add(np.ones(3), np.arange(3))`  # ??? |
| `-` or `np.subtract(x1, x2)` | ??? | `np.subtract(np.arange(3), 1)`  # ??? |
| `*` or `np.multiply(x1, x2)` | ??? | `np.multiply(2, np.arange(3))`  # ??? |
| `/` or `np.divide(x1, x2)` | ??? | `np.divide(np.arange(1,4), 2)`  # ??? |
| `**` or `np.power(x1, x2)` | ??? | `np.power(2, np.arange(3))`  # ??? |
| `np.sqrt(x)` | ??? | `np.sqrt(np.array([4,9,16]))`  # ??? |
| `np.exp(x)` | ??? | `np.exp(np.array([0,1]))`  # ??? |
| `np.log(x)` | ??? | `np.log(np.exp(144))`  # ??? |
| `np.sin(x)`, `np.cos(x)`, `np.tan(x)` | Trigonometric functions or angles. | `np.sin(np.pi/2)`  # ??? |
| `np.abs(x)` | Absolute value. | `np.abs(np.array([-1,0,1]))`  # ??? |
| `np.round(a, decimals=0)` | Rounding or precision control. | `np.round(3.14159, 2)`  # ??? |
| `np.clip(a, a_min, a_max)` | Limits values to a range. | `np.clip(np.arange(5), 1, 3)`  # ??? |
| `np.cumsum(a, axis=None)` | Cumulative sum totals. | `???`  # array([0,1,3,6]) |
| `np.diff(a, n=1, axis=-1)` | ??? | `???`  # array([3,5]) |

<details>
<summary>Reveal answer</summary>

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `x1 + x2` or `np.add(x1, x2)` | Element-wise addition. | `np.add(np.ones(3), np.arange(3))`  # array([1,2,3]) |
| `-` or `np.subtract(x1, x2)` | Element-wise subtraction. | `np.subtract(np.arange(3), 1)`  # array([-1,0,1]) |
| `*` or `np.multiply(x1, x2)` | Element-wise multiplication. | `np.multiply(2, np.arange(3))`  # array([0,2,4]) |
| `/` or `np.divide(x1, x2)` | Element-wise division. | `np.divide(np.arange(1,4), 2)`  # array([0.5,1.,1.5]) |
| `**` or `np.power(x1, x2)` | Element-wise exponentiation. | `np.power(2, np.arange(3))`  # array([1,2,4]) |
| `np.sqrt(x)` | Square root. | `np.sqrt(np.array([4,9,16]))`  # array([2,3,4]) |
| `np.exp(x)` | Exponential. | `np.exp(np.array([0,1]))`  # array([1., 2.71828183]) |
| `np.log(x)` | Natural log. | `np.log(np.exp(144))`  # 144.0 |
| `np.sin(x)`, `np.cos(x)`, `np.tan(x)` | Trigonometric functions or angles. | `np.sin(np.pi/2)`  # 1.0 |
| `np.abs(x)` | Absolute value. | `np.abs(np.array([-1,0,1]))`  # array([1,0,1]) |
| `np.round(a, decimals=0)` | Rounding or precision control. | `np.round(3.14159, 2)`  # 3.14 |
| `np.clip(a, a_min, a_max)` | Limits values to a range. | `np.clip(np.arange(5), 1, 3)`  # array([1,1,2,3,3]) |
| `np.cumsum(a, axis=None)` | Cumulative sum totals. | `np.cumsum(np.arange(4))`  # array([0,1,3,6]) |
| `np.diff(a, n=1, axis=-1)` | Discrete differences. | `np.diff(np.array([1,4,9]))`  # array([3,5]) |

</details>

### 6. Linear Algebra

Functions for matrix operations, solving systems of equations, decompositions, etc.

Use the official documentation of NumPy: <https://numpy.org/doc/stable/reference/arrays.ndarray.html#the-n-dimensional-array-ndarray>, to fill in the missing parts (marked with `???`) below.

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `np.dot(a, b)` or `a @ b` | ??? | `np.dot(np.eye(2), np.array([1,2]))`  # ??? |
| `np.matmul(a, b)` | ??? | `np.matmul(np.ones((2,2)), np.ones((2,1)))`  # ??? |
| `np.linalg.inv(a)` | ??? | `np.linalg.inv(np.array([[1,2],[3,4]]))`  # `array([[-2. ,  1. ], [ 1.5, -0.5]])` |
| `np.linalg.det(a)` | Determinant, for invertibility checks. | `np.linalg.det(np.eye(3))`  # ??? |
| `???` | Eigenvalues and vectors. | `???` # ??? |
| `np.linalg.solve(a, b)` | ??? | `np.linalg.solve(np.array([[1,1],[1,2]]), np.array([3,5]))`  # ??? |
| `np.linalg.norm(x, ord=None)` | ??? | `np.linalg.norm(np.array([3,4]))`  # ??? |
| `np.cross(a, b)` | ??? | `np.cross([1,0,0], [0,1,0])`  # ??? |
| `np.trace(a)` | ??? | `np.trace(np.eye(3))`  # ??? |

<details>
<summary>Reveal answer</summary>

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `np.dot(a, b)` or `a @ b` | Matrix multiplication. | `np.dot(np.eye(2), np.array([1,2]))`  # array([1,2]) |
| `np.matmul(a, b)` | Similar to dot but for multi-dim, broadcasting-aware. | `np.matmul(np.ones((2,2)), np.ones((2,1)))`  # `[[2],[2]]` |
| `np.linalg.inv(a)` | Matrix inverse, for solving linear systems. | `np.linalg.inv(np.array([[1,2],[3,4]]))`  # `array([[-2. ,  1. ], [ 1.5, -0.5]])` |
| `np.linalg.det(a)` | Determinant, for invertibility checks. | `np.linalg.det(np.eye(3))`  # 1.0 |
| `np.linalg.eig(a)` | Eigenvalues and vectors. | `eigen_values, eigen_vectors = np.linalg.eig(np.array([[1,0],[0,2]]))` # eigen_values=`array([1., 2.])` eigen_vectors=`array([[1., 0.], [0., 1.]])` |
| `np.linalg.solve(a, b)` | Solves Ax = b. | `np.linalg.solve(np.array([[1,1],[1,2]]), np.array([3,5]))`  # array([1,2]) |
| `np.linalg.norm(x, ord=None)` | Vector or matrix norms, for distances. | `np.linalg.norm(np.array([3,4]))`  # 5.0 (Euclidean) |
| `np.cross(a, b)` | Cross product, for vectors in 3D (produces a third vector perpendicular to both original vectors). | `np.cross([1,0,0], [0,1,0])`  # array([0,0,1]) |
| `np.trace(a)` | Return the sum along diagonals of the array. | `np.trace(np.eye(3))`  # 3.0 |

</details>

### 7. Statistical Functions

Aggregate statistics on arrays, for data summarization and analysis.

Use the official documentation of NumPy: <https://numpy.org/doc/stable/reference/arrays.ndarray.html#the-n-dimensional-array-ndarray>, to fill in the missing parts (marked with `???`) below.

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `np.mean(a, axis=None)` | ??? | `np.mean(np.array([1,3,2,2,2,200]))`  # ??? |
| `np.median(a, axis=None)` | ??? | `np.median(np.array([1,3,2,2,2,200]))`  # ??? |
| `np.std(a, axis=None)` | ??? | `np.std(np.array([1,2,3]))`  # ≈0.816 |
| `np.var(a, axis=None)` | ??? | `???`  # 2.0 |
| `np.sum(a, axis=None)` | ??? | `???`  # 5.0 |
| `np.prod(a, axis=None)` | ??? | `???`  # 6 |
| `np.min(a, axis=None)`, `np.max(a, axis=None)` | ??? | `???`  # ??? |
| `np.argmin(a, axis=None)`, `np.argmax(a, axis=None)` | ??? | `???`  # ??? |
| `np.percentile(a, q, axis=None)` | ??? | `???`  # 50.0 |
| `np.corrcoef(x, y=None)` | Correlation coefficients, for relationships. | `np.corrcoef(np.arange(3), np.arange(3)[::-1])`  # Correlation matrix |
| `np.histogram(a, bins=10)` | ??? | `???` |
| `np.bincount(x, weights=None)` | ??? | `np.bincount([0,1,1,2])`  # ??? |

<details>
<summary>Reveal answer</summary>

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `np.mean(a, axis=None)` | Average value. | `np.mean(np.array([1,3,2,2,2,200]))`  # 35.0 |
| `np.median(a, axis=None)` | Median, robust to outliers. | `np.median(np.array([1,3,2,2,2,200]))`  # 2.0 |
| `np.std(a, axis=None)` | Standard deviation, for variability. | `np.std(np.array([1,2,3]))`  # ≈0.816 |
| `np.var(a, axis=None)` | Variance, for spread. | `np.var(np.arange(5))`  # 2.0 |
| `np.sum(a, axis=None)` | Total sum. | `np.sum(np.ones(5))`  # 5.0 |
| `np.prod(a, axis=None)` | Product of elements. | `np.prod(np.arange(1,4))`  # 6 |
| `np.min(a, axis=None)`, `np.max(a, axis=None)` | Min/max values. | `np.min(np.array([-1,0,1]))`  # -1 |
| `np.argmin(a, axis=None)`, `np.argmax(a, axis=None)` | Indices of min/max. | `np.argmax([1,3,2])`  # 1 |
| `np.percentile(a, q, axis=None)` | Percentiles, for quantiles. | `np.percentile(np.arange(101), 50)`  # 50.0 |
| `np.corrcoef(x, y=None)` | Correlation coefficients, for relationships. | `np.corrcoef(np.arange(3), np.arange(3)[::-1])`  # Correlation matrix |
| `np.histogram(a, bins=10)` | Histogram computation, for distributions. | `hist, bins = np.histogram(np.random.randn(100), 5)` |
| `np.bincount(x, weights=None)` | Counts occurrences. | `np.bincount([0,1,1,2])`  # array([1,2,1]) |

</details>

### 8. Broadcasting

Broadcasting allows operations on arrays of different shapes, solving mismatched dimension problems without loops.

- **Problem Solved**: Apply an operation without explicit replication, e.g., adding a scalar to an array or a row to a matrix.
- **Examples**:

```python
arr = np.arange(6).reshape(2,3)
arr
```

```console
array([[0, 1, 2],
       [3, 4, 5]])
```

```python
scalar_add = arr + 5 # <- This is broadcasting: adds 5 to all elements. Under the hood: arr + np.full((2,3), 5)
scalar_add
```

```console
array([[ 5,  6,  7],
       [ 8,  9, 10]])
```

```python
row_add = arr + np.array([10,20,30])  # Broadcasts row to matrix
row_add
```

```console
array([[10, 21, 32],
       [13, 24, 35]])
```

- **Broadcasing semantics**: Broadcasting does not work with all types of arrays.

Two arrays are *broadcastable* if the following two rules hold:

1. Each array has at least one dimension.
2. When iterating over the dimension sizes, starting at the trailing/right-most dimension, the dimension sizes must either be equal, one of them is `1`, or one of them does not exist.

```python
a = np.array([[1,2,3],[4,5,6]]) # dimension/shape: (2, 3)
b = np.array([1,2,3])           # dimension/shape: (1, 3)
a + b                           # Rule 1: ok. Rule 2: ok, since 3 = 3 and b's first dimension is 1.
# [[2,4,6],
#  [5,7,9]]
```

Explain whether x and y broadcastable:

```python
x = np.ones((5, 7, 3))
y = np.ones((5, 7, 3))
```

<details>
<summary>Reveal answer</summary>

same shapes are always broadcastable

```python
x + y
```

```console
array([[[2., 2., 2.],
        [2., 2., 2.],
        ...
        [2., 2., 2.],
        [2., 2., 2.]]])
```

</details>

Explain whether x and y broadcastable:

```python
x=np.ones((0,))
y=np.ones((2,2))
```

<details>
<summary>Reveal answer</summary>

x and y are not broadcastable, because x does not have at least 1 dimension

```python
x + y
```

```console
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (0,) (2,2)
```

</details>

Explain whether x and y broadcastable:

```python
x=np.ones((5,3,4,1))
y=np.ones((3,1,1))
```

<details>
<summary>Reveal answer</summary>

x and y are broadcastable, since the trailing dimensions line up:

- 1st trailing dimension: both have size 1;
- 2nd trailing dimension: y has size 1;
- 3rd trailing dimension: x size == y size;
- 4th trailing dimension: y dimension doesn't exist.

```python
x + y
```

```console
array([[[[2.],
         [2.],
         ...
         [2.],
         [2.]]]])
```

</details>

Explain whether x and y broadcastable:

```python
x=np.ones((5,2,4,1))
y=np.ones((3,1,1))
```

<details>
<summary>Reveal answer</summary>

x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3

```python
x + y
```

```console
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (5,2,4,1) (3,1,1)
```

</details>

### 9. Random Number Generation

Functions for simulations, sampling, or initialization with (pseudo-) random values.

Use the official documentation of NumPy: <https://numpy.org/doc/stable/reference/arrays.ndarray.html#the-n-dimensional-array-ndarray>, to fill in the missing parts (marked with `???`) below.

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `np.random.rand(d0, d1, ...)` | ??? | `???`  # 2x2 random matrix |
| `np.random.randn(d0, d1, ...)` | ??? | `???` |
| `np.random.randint(low, high=None, size=None)` | ??? | `???`  # 5 ints between 0-9 |
| `np.random.choice(a, size=None, replace=True)` | ??? | `???` |
| `np.random.shuffle(x)` | ??? | `arr = np.arange(5); np.random.shuffle(arr); arr` |
| `np.random.uniform(low=0.0, high=1.0, size=None)` | Uniform distribution. | `???` |
| `np.random.normal(loc=0.0, scale=1.0, size=None)` | Normal distribution. | `???` |

<details>
<summary>Reveal answer</summary>

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `np.random.rand(d0, d1, ...)` | Uniform [0,1) random numbers. | `np.random.rand(2,2)`  # 2x2 random matrix |
| `np.random.randn(d0, d1, ...)` | Standard normal distribution. | `np.random.randn(3)` |
| `np.random.randint(low, high=None, size=None)` | Random integers. | `np.random.randint(0, 10, 5)`  # 5 ints between 0-9 |
| `np.random.choice(a, size=None, replace=True)` | Samples from array. | `np.random.choice(['a','b','c'], 2)` |
| `np.random.shuffle(x)` | Shuffles array in-place. | `arr = np.arange(5); np.random.shuffle(arr); arr` |
| `np.random.uniform(low=0.0, high=1.0, size=None)` | Uniform distribution. | `np.random.uniform(-1,1,3)` |
| `np.random.normal(loc=0.0, scale=1.0, size=None)` | Normal distribution. | `np.random.normal(0, 2, 5)` |

</details>

### 10. File I/O and Miscellaneous

Saving/loading arrays, and other utilities.

Use the official documentation of NumPy: <https://numpy.org/doc/stable/reference/arrays.ndarray.html#the-n-dimensional-array-ndarray>, to fill in the missing parts (marked with `???`) below.

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `np.savetxt(fname, X, delimiter=' ')` | Saves to text file, for human-readable output. | `np.savetxt('data.txt', np.arange(5))` |
| `np.loadtxt(fname, delimiter=None)` | Loads from text, for importing CSV-like data. | `txt_data = np.loadtxt('data.txt')` |
| `np.copy(a)` | Deep copy of array. | `np.copy(arr)` |
| `np.sort(a, axis=-1, kind='quicksort')` | Sorts array, for ordering data. | `???`  # array([1,2,3]) |
| `np.argsort(a, axis=-1)` | ??? | `np.argsort([3,1,2])`  # ??? |
| `np.searchsorted(a, v)` | ??? | `???`  # 2 |
| `np.all(a, axis=None)` | ??? | `np.all(np.array([True, True]))`  # ??? |
| `np.any(a, axis=None)` | ??? | `np.any(np.array([False, True]))`  # ??? |
| `np.isnan(a)` | ??? | `???`  # [False, True] |
| `np.isinf(a)` | ??? | `???`  # [False, True] |

<details>
<summary>Reveal answer</summary>

| Function | Problem Solved | Example |
|----------|----------------|---------|
| `np.savetxt(fname, X, delimiter=' ')` | Saves to text file, for human-readable output. | `np.savetxt('data.txt', np.arange(5))` |
| `np.loadtxt(fname, delimiter=None)` | Loads from text, for importing CSV-like data. | `txt_data = np.loadtxt('data.txt')` |
| `np.copy(a)` | Deep copy of array. | `np.copy(arr)` |
| `np.sort(a, axis=-1, kind='quicksort')` | Sorts array, for ordering data. | `np.sort([3,1,2])`  # array([1,2,3]) |
| `np.argsort(a, axis=-1)` | Indices that would sort, for indirect sorting. | `np.argsort([3,1,2])`  # array([1,2,0]) |
| `np.searchsorted(a, v)` | Finds insertion points for sorted arrays. | `np.searchsorted([1,3,5], 4)`  # 2 |
| `np.all(a, axis=None)` | Checks if all elements are true, for conditions. | `np.all(np.array([True, True]))`  # True |
| `np.any(a, axis=None)` | If any true, for existence checks. | `np.any(np.array([False, True]))`  # True |
| `np.isnan(a)` | Detects NaNs, for data cleaning. | `np.isnan(np.array([1, np.nan]))`  # [False, True] |
| `np.isinf(a)` | Detects infinities. | `np.isinf(np.array([1, np.inf]))`  # [False, True] |

</details>

`numpy` is great for doing vector arithmetic operations. If you compare its functionality with regular Python lists, however, some things have changed:

- `numpy` arrays cannot contain elements with different types;
- the typical arithmetic operators, such as `+`, `-`, `*` and `/` have a different meaning for regular Python lists and `numpy` arrays.

Four lines of code have been provided for you:

A. `np.array([True, 1, 2, 3, 4, False])`
B. `np.array([4, 3, 0]) + np.array([0, 2, 2])`
C. `np.array([1, 1, 2]) + np.array([3, 4, -1])`
D. `np.array([0, 1, 2, 3, 4, 5])`

Which one of the above four lines is equivalent to the following expression?

```python
np.array([True, 1, 2]) + np.array([3, 4, False])
```

<details>
<summary>Reveal answer</summary>

$B$.

</details>

## Pandas

Pandas is a high-level Python library for data manipulation and analysis, built on NumPy.

It excels at handling tabular data with mixed data types (e.g., strings, floats) in structures like Series (1D labeled array) and DataFrame (2D table with labeled axes).

### 0. Importing

By convention, `pandas` is imported with the alias `pd`:

```python
import pandas as pd
import numpy as np  # Often used with Pandas
```

### 1. Introduction to DataFrames

Tabular data, like spreadsheets, consists of rows (observations) and columns (variables). For example, in a chemical plant, you might have temperature measurements:

| Observation | Temperature | Date       | Location |
|-------------|-------------|------------|----------|
| 1           | 25.5        | 2025-01-01 | Plant A  |
| 2           | 26.2        | 2025-01-02 | Plant B  |

Pandas stores such data in a **DataFrame**, which supports:

- Labeled rows and columns.
- Mixed data types (e.g., strings, numbers).
- Efficient handling of large datasets.

For example, consider the BRICS dataset:

```python
data = {
    'country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'capital': ['Brasilia', 'Moscow', 'New Delhi', 'Beijing', 'Pretoria'],
    'area': [8.516, 17.10, 3.286, 9.597, 1.221],
    'population': [200.4, 143.5, 1252, 1357, 52.98]
}
df_brics = pd.DataFrame(data, index=['BR', 'RU', 'IN', 'CH', 'SA'])
```

```console
         country    capital    area  population
BR        Brazil   Brasilia   8.516      200.40
RU        Russia     Moscow  17.100      143.50
IN         India  New Delhi   3.286     1252.00
CH         China    Beijing   9.597     1357.00
SA  South Africa   Pretoria   1.221       52.98
```

### 2. Data Structures and Creation

Create Series (1D) and DataFrames (2D) from various sources.

Use the official Pandas documentation: <https://pandas.pydata.org/docs/reference/index.html>, to fill in the missing parts (marked with `???`) below.

| Function | Problem Solved | Example |
|----------------|----------------|---------|
| `pd.Series(data, index=None)` | Creates a 1D labeled array. | `???`  # Creates a series with values [1, 2, 3] and index ['a', 'b', 'c']  |
| `pd.DataFrame(data, index=None, columns=None)` | Creates a 2D table from dicts/lists. | `???`  # Creates a DataFrame with columns 'A', 'B' and values 1, 2 for A, and 3, 4 for B. |
| `???` | Loads CSV data into a DataFrame. | `???`  # Reads in the data in 'brics.csv' and uses the first column as index. |
| `???` | Creates DataFrame from list of tuples. | `???`  # Creates a DataFrame with columns 'num', 'letter', and values 1, 2 for 'num' and 'x', 'y' for 'letter' |

<details>
<summary>Reveal answer</summary>

| Function | Problem Solved | Example |
|----------------|----------------|---------|
| `pd.Series(data, index=None)` | Creates a 1D labeled array. | `pd.Series([1, 2, 3], index=['a', 'b', 'c'])`  # Creates a series with values [1, 2, 3] and index ['a', 'b', 'c'] |
| `pd.DataFrame(data, index=None, columns=None)` | Creates a 2D table from dicts/lists. | `pd.DataFrame({'A': [1, 2], 'B': [3, 4]})`  # Creates a DataFrame with columns 'A', 'B' and values 1, 2 for A, and 3, 4 for B. |
| `pd.read_csv(file, delimiter=',')` | Loads CSV data into a DataFrame. | `pd.read_csv('brics.csv', index_col=0)`  # Reads in the data in 'brics.csv' and uses the first column as index. |
| `pd.DataFrame.from_records(data)` | Creates DataFrame from list of tuples. | `pd.DataFrame.from_records([(1, 'x'), (2, 'y')], columns=['num', 'letter'])`  # Creates a DataFrame with columns 'num', 'letter', and values 1, 2 for 'num' and 'x', 'y' for 'letter' |

</details>

### 3. Data Inspection

Inspect DataFrame metadata and summaries.

Use the official Pandas documentation: <https://pandas.pydata.org/docs/reference/index.html>, to fill in the missing parts (marked with `???`) below.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
```

| Property | Problem Solved | Example |
|--------|----------------|---------|
| `.shape` | ??? | `df.shape`  # (3, 2) |
| `.info()` | ??? | `df.info()` |
| `.describe()` | ??? | `df.describe()` |
| `.describe(include='object')` | ??? | `df.describe(include='object')` |
| `.head(n=5)` | ??? | `df.head(2)` # Shows the first ??? rows. |
| `.dtypes` | ??? | `df.dtypes` |
| `.columns` | ??? | `df.columns` |

<details>
<summary>Reveal answer</summary>

| Property | Problem Solved | Example |
|--------|----------------|---------|
| `.shape` | Returns dimensions (rows, columns). | `df.shape`  # (3, 2) |
| `.info()` | Shows DataFrame structure and types. | `df.info()` |
| `.describe()` | Summarizes numeric columns. | `df.describe()` |
| `.describe(include='object')` | Summarizes string columns. | `df.describe(include='object')` |
| `.head(n=5)` | Views first n rows. | `df.head(2)` # Shows the first 2 rows. |
| `.dtypes` | Returns column data types. | `df.dtypes` |
| `.columns` | Returns the names of the columns. | `df.columns` |

</details>

### 4. Data Selection and Indexing

Access data using labels (`loc`), positions (`iloc`), or brackets (`[]`).

Use the official Pandas documentation: <https://pandas.pydata.org/docs/reference/index.html>, to fill in the missing parts (marked with `???`) below.

```python
df_brics = pd.DataFrame({
    'country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'capital': ['Brasilia', 'Moscow', 'New Delhi', 'Beijing', 'Pretoria'],
    'area': [8.516, 17.10, 3.286, 9.597, 1.221],
    'population': [200.4, 143.5, 1252, 1357, 52.98]
}, index=['BR', 'RU', 'IN', 'CH', 'SA'])
```

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `df['column']` | ??? | `???`  # Series(['Brazil', 'Russia', 'India', 'China', 'South Africa'], index=['BR', 'RU', 'IN', 'CH', 'SA']) |
| `df[['col1', 'col2']]` | ??? | `???`  # DataFrame with 'country', 'capital' columns |
| `df.loc[label]` | ??? | `???`  # Series(country='Russia', capital='Moscow', area=17.1, population=143.5) |
| `df.iloc[integer]` | ??? | `???`  # Series(country='Russia', capital='Moscow', area=17.1, population=143.5) |
| `df.loc[:, cols]` | ??? | `???`  # DataFrame with all rows, 'country' and 'capital' columns |
| `df.at[label, column]` | ??? | `???`  # 'Moscow' |
| `df.iat[integer, integer]` | ??? | `???`  # 'Russia' |

<details>
<summary>Reveal answer</summary>

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `df['column']` | Selects column as Series. | `df_brics['country']`  # Series(['Brazil', 'Russia', 'India', 'China', 'South Africa'], index=['BR', 'RU', 'IN', 'CH', 'SA']) |
| `df[['col1', 'col2']]` | Selects columns as DataFrame. | `df_brics[['country', 'capital']]`  # DataFrame with 'country', 'capital' columns |
| `df.loc[label]` | Label-based row/column access. | `df_brics.loc['RU']`  # Series(country='Russia', capital='Moscow', area=17.1, population=143.5) |
| `df.iloc[integer]` | Position-based row/column access. | `df_brics.iloc[1]`  # Series(country='Russia', capital='Moscow', area=17.1, population=143.5) |
| `df.loc[:, cols]` | Selects all rows, specific columns. | `df_brics.loc[:, ['country', 'capital']]`  # DataFrame with all rows, 'country' and 'capital' columns |
| `df.at[label, column]` | Fast scalar access by label. | `df_brics.at['RU', 'capital']`  # 'Moscow' |
| `df.iat[integer, integer]` | Fast scalar access by position. | `df_brics.iat[1, 0]`  # 'Russia' |

</details>

### 5. Filtering Data

Filter rows based on conditions.

Use the official Pandas documentation: <https://pandas.pydata.org/docs/reference/index.html>, to fill in the missing parts (marked with `???`) below.

```python
df_brics = pd.DataFrame({
    'country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'capital': ['Brasilia', 'Moscow', 'New Delhi', 'Beijing', 'Pretoria'],
    'area': [8.516, 17.10, 3.286, 9.597, 1.221],
    'population': [200.4, 143.5, 1252, 1357, 52.98]
}, index=['BR', 'RU', 'IN', 'CH', 'SA'])
```

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `df[condition]` | ??? | `df_brics[df_brics['area'] > 8]`  # DataFrame with rows 'BR', 'RU', 'CH' |
| `df[(condition1) & (condition2)]` | ??? | `df_brics[(df_brics['area'] >= 8) & (df_brics['area'] <= 10)]`  # DataFrame with rows 'BR', 'CH' |
| `???` | Filters values in a range. | `???`  # DataFrame with rows 'BR', 'CH' |
| `df.loc[condition, cols]` | ??? | `???`  # Series(['Brazil', 'Russia', 'China']) |

<details>
<summary>Reveal answer</summary>

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `df[condition]` | Filters rows by condition. | `df_brics[df_brics['area'] > 8]`  # DataFrame with rows 'BR', 'RU', 'CH' |
| `df[condition1 & condition2]` | Filters with multiple conditions. | `df_brics[(df_brics['area'] >= 8) & (df_brics['area'] <= 10)]`  # DataFrame with rows 'BR', 'CH' |
| `df['col'].between(left, right)` | Filters values in a range. | `df_brics[df_brics['area'].between(8, 10)]`  # DataFrame with rows 'BR', 'CH' |
| `df.loc[condition, cols]` | Filters rows and selects columns. | `df_brics.loc[df_brics['area'] > 8, 'country']`  # Series(['Brazil', 'Russia', 'China']) |

</details>

### 6. Data Manipulation

Reshape, clean, or transform data.

Use the official Pandas documentation: <https://pandas.pydata.org/docs/reference/index.html>, to fill in the missing parts (marked with `???`) below.

```python
df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, 5, 6]})
df_brics = pd.DataFrame({
    'country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'capital': ['Brasilia', 'Moscow', 'New Delhi', 'Beijing', 'Pretoria'],
    'area': [8.516, 17.10, 3.286, 9.597, 1.221],
    'population': [200.4, 143.5, 1252, 1357, 52.98]
}, index=['BR', 'RU', 'IN', 'CH', 'SA'])
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [4, 5, 6]})
```

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `???` | Removes rows/columns with NaNs. | `???`  # DataFrame with row 2 removed |
| `df.fillna(value)` | ??? | `???`  # DataFrame with NaN replaced by 0 |
| `???` | Replaces specific values. | `???`  # DataFrame with 1 replaced by 10 |
| `df.sort_values(by)` | ??? | `???`  # DataFrame sorted by 'A' ascending |
| `???` | Renames columns. | `???`  # DataFrame with 'A' renamed to 'X' |
| `df.apply(func, axis=0)` | ??? | `???`  # Series([1, 4, nan]) |
| `df['col'].apply(func)` | ??? | `???`  # Series([6, 6, 5, 5, 12]) |
| `df.applymap(func)` | Applies function element-wise to entire DataFrame. | `???`  # DataFrame with non-NaN values doubled |
| `df['col'].map(func)` | Applies function or mapping to Series elements. | `???`  # Series(['BR', 'RU', NaN, NaN, NaN]) |
| `df.iterrows()` | ??? | `???`  # Prints: Brasilia, Moscow, New Delhi, Beijing, Pretoria |
| ??? | Concatenates DataFrames along axis. | `???`  # DataFrame with the rows of df1 and df2 stacked vertically |
| `df1.merge(df2, how='inner')` | Merges on key(s). | `???`  # DataFrame with rows for 'B', 'C' |
| `df1.join(df2, lsuffix='_left')` | Joins on index. | `???`  # DataFrame with matched rows with suffixed matching columns |

<details>
<summary>Reveal answer</summary>

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `df.dropna(axis=0)` | Removes rows/columns with NaNs. | `df.dropna()`  # DataFrame with row 2 removed |
| `df.fillna(value)` | Fills NaNs with a value. | `df.fillna(0)`  # DataFrame with NaN replaced by 0 |
| `df.replace(to_replace, value)` | Replaces specific values. | `df.replace(1, 10)`  # DataFrame with 1 replaced by 10 |
| `df.sort_values(by)` | Sorts by column(s). | `df.sort_values('A')`  # DataFrame sorted by 'A' ascending |
| `df.rename(columns=dict)` | Renames columns. | `df.rename(columns={'A': 'X'})`  # DataFrame with 'A' renamed to 'X' |
| `df.apply(func, axis=0)` | Applies function along axis (e.g., column). | `df.apply(np.sum)`  # Series(A=3.0, B=15.0) |
| `df['col'].apply(func)` | Applies function element-wise to column. | `df_brics['country'].apply(len)`  # Series([6, 6, 5, 5, 12]) |
| `df.applymap(func)` | Applies function element-wise to entire DataFrame. | `df.applymap(lambda x: x*2 if pd.notna(x) else x)`  # DataFrame with non-NaN values doubled |
| `df['col'].map(func)` | Applies function or mapping to Series elements. | `df_brics['country'].map({'Brazil': 'BR', 'Russia': 'RU'})`  # Series(['BR', 'RU', NaN, NaN, NaN]) |
| `df.iterrows()` | Iterates over rows as (index, Series) pairs. | `for lab, row in df_brics.iterrows(): print(row['capital'])`  # Prints: Brasilia, Moscow, New Delhi, Beijing, Pretoria |
| `pd.concat([df1, df2], axis=0)` | Concatenates DataFrames along axis. | `pd.concat([df1, df2])`  # DataFrame with the rows of df1 and df2 stacked vertically |
| `df1.merge(df2, how='inner')` | Merges on key(s). | `df1.merge(df2, on='key', how='inner')`  # DataFrame with rows for 'B', 'C' |
| `df1.join(df2, lsuffix='_left')` | Joins on index. | `df1.set_index('key').join(df2.set_index('key'), lsuffix='_left').reset_index()`  # DataFrame with matched rows with suffixed matching columns |

</details>

### 7. Statistical and Mathematical Operations

Perform calculations on DataFrames/Series.

Use the official Pandas documentation: <https://pandas.pydata.org/docs/reference/index.html>, to fill in the missing parts (marked with `???`) below.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
```

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `???` | Computes column mean. | `???`  # Series(A=2.0, B=5.0) |
| `???` | Computes column standard deviation. | `???`  # Series(A=1.0, B=1.0) |
| `???` | Computes column sum. | `???`  # Series(A=6, B=15) |
| `???` | Computes cumulative sum. | `???`  # DataFrame `[[1,4],[3,9],[6,15]]` |
| `???` | Computes correlation matrix. | `???`  # 2x2 DataFrame with correlation coefficients |
| `???` | Element-wise arithmetic. | `???`  # DataFrame with 10 added to each element |

<details>
<summary>Reveal answer</summary>

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `df.mean(axis=0)` | Computes column mean. | `df.mean()`  # Series(A=2.0, B=5.0) |
| `df.std(axis=0)` | Computes column standard deviation. | `df.std()`  # Series(A=1.0, B=1.0) |
| `df.sum(axis=0)` | Computes column sum. | `df.sum()`  # Series(A=6, B=15) |
| `df.cumsum(axis=0)` | Computes cumulative sum. | `df.cumsum()`  # DataFrame `[[1,4],[3,9],[6,15]]` |
| `df.corr()` | Computes correlation matrix. | `df.corr()`  # 2x2 DataFrame with correlation coefficients |
| `df + scalar` | Element-wise arithmetic. | `df + 10`  # DataFrame with 10 added to each element |

</details>

### 8. Handling Missing Data

Detect and handle missing values (`NaN`, `None`).

Use the official Pandas documentation: <https://pandas.pydata.org/docs/reference/index.html>, to fill in the missing parts (marked with `???`) below.

```python
df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
```

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `???` | Detects missing values. | `???`  # DataFrame with `True` where values are `NaN` |
| `???` | Detects non-missing values. | `???`  # DataFrame with `True` where values are not `NaN` |
| `???` | Fills missing values. | `???`  # DataFrame with `NaNs` replaced by `0` |
| `???` | Removes rows with NaNs. | `???`  # DataFrame with only row `0` (no `NaNs`) |

<details>
<summary>Reveal answer</summary>

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `df.isna()` | Detects missing values. | `df.isna()`  # DataFrame with `True` where values are `NaN` |
| `df.notna()` | Detects non-missing values. | `df.notna()`  # DataFrame with `True` where values are not `NaN` |
| `df.fillna(value)` | Fills missing values. | `df.fillna(0)`  # DataFrame with `NaNs` replaced by `0` |
| `df.dropna()` | Removes rows with NaNs. | `df.dropna()`  # DataFrame with only row `0` (no `NaNs`) |

</details>

### 9. Grouping and Aggregation

Group data and aggregate.

Use the official Pandas documentation: <https://pandas.pydata.org/docs/reference/index.html>, to fill in the missing parts (marked with `???`) below.

```python
df = pd.DataFrame({'group': ['X', 'X', 'Y'], 'value': [1, 2, 3]})
```

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `???` | Computes group means. | `???`  # DataFrame: X=1.5, Y=3.0 for 'value' |
| `???` | Counts group occurrences. | `???`  # Series(X=2, Y=1) |
| `???` | Applies custom aggregation. | `???`  # DataFrame: X=3, Y=3 for 'value' |

<details>
<summary>Reveal answer</summary>

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `df.groupby(by=column).mean()` | Computes group means. | `df.groupby('group').mean()`  # DataFrame: X=1.5, Y=3.0 for 'value' |
| `df.groupby(by=column).size()` | Counts group occurrences. | `df.groupby('group').size()`  # Series(X=2, Y=1) |
| `df.groupby(by=column).agg(func)` | Applies custom aggregation. | `df.groupby('group').agg({'value': 'sum'})`  # DataFrame: X=3, Y=3 for 'value' |

</details>

### 10. File I/O and Miscellaneous

Save/load DataFrames and other utilities.

Use the official Pandas documentation: <https://pandas.pydata.org/docs/reference/index.html>, to fill in the missing parts (marked with `???`) below.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
```

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `???` | Saves DataFrame to CSV. | `???`  # Writes DataFrame to 'data.csv' with index |
| `???` | Loads CSV into DataFrame. | `???`  # Loads 'data.csv', first column as index |
| `???` | Creates deep copy. | `???`  # New DataFrame with same data |
| `df.pivot_table(values, index, columns)` | Creates pivot table. | `df.pivot_table(values='A', index='B')`  # Pivot table with 'A' values, 'B' as index |
| `df.melt(id_vars)` | Unpivots to long format. | `df.melt(id_vars='A')`  # DataFrame with 'A' as id, 'B' as variable-value pairs |

<details>
<summary>Reveal answer</summary>

| Method | Problem Solved | Example |
|--------|----------------|---------|
| `df.to_csv(file, index=True)` | Saves DataFrame to CSV. | `df.to_csv('data.csv')`  # Writes DataFrame to 'data.csv' with index |
| `pd.read_csv(file)` | Loads CSV into DataFrame. | `pd.read_csv('data.csv', index_col=0)`  # Loads 'data.csv', first column as index |
| `df.copy()` | Creates deep copy. | `df.copy()`  # New DataFrame with same data |
| `df.pivot_table(values, index, columns)` | Creates pivot table. | `df.pivot_table(values='A', index='B')`  # Pivot table with 'A' values, 'B' as index |
| `df.melt(id_vars)` | Unpivots to long format. | `df.melt(id_vars='A')`  # DataFrame with 'A' as id, 'B' as variable-value pairs |

</details>

## Matplotlib

The better you understand your data, the better you'll be able to extract insights. And once you've found those insights, again, you'll need visualization to be able to share your valuable insights with other people.

![w01_matplotlib.png](./assets/w01_matplotlib.png "w01_matplotlib.png")

There are many visualization packages in python, but the mother of them all, is `matplotlib`. You will need its subpackage `pyplot`. By convention, this subpackage is imported as `plt`:

```python
import matplotlib.pyplot as plt
```

### Line plot

Let's try to gain some insights in the evolution of the world population. To plot data as a **line chart**, we call `plt.plot` and use our two lists as arguments. The first argument corresponds to the horizontal axis, and the second one to the vertical axis.

```python
year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]

# "plt.plot" creates the plot, but does not display it
plt.plot(year, pop)

# "plt.show" displays the plot
plt.show()
```

You'll have to call `plt.show()` explicitly because you might want to add some extra information to your plot before actually displaying it, such as titles and label customizations.

As a result we get:

![w01_matplotlib_result.png](./assets/w01_matplotlib_result.png "w01_matplotlib_result.png")

We see that:

- the years are indeed shown on the horizontal axis;
- the populations on the vertical axis;
- this type of plot is great for plotting a time scale along the x-axis and a numerical feature on the y-axis.

There are four data points, and Python draws a line between them.

![w01_matplotlib_edited.png](./assets/w01_matplotlib_edited.png "w01_matplotlib_edited.png")

In 1950, the world population was around 2.5 billion. In 2010, it was around 7 billion.

> **Insight:** The world population has almost tripled in sixty years.
>
> **Note:** If you pass only one argument to `plt.plot`, Python will know what to do and will use the index of the list to map onto the `x` axis, and the values in the list onto the `y` axis.

### Scatter plot

We can reuse the code from before and just swap `plt.plot(...)` with `plt.scatter(...)`:

```python
year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]

# "plt.plot" creates the plot, but does not display it
plt.scatter(year, pop)

# "plt.show" displays the plot
plt.show()
```

![w01_matplotlib_scatter.png](./assets/w01_matplotlib_scatter.png "w01_matplotlib_scatter.png")

The resulting scatter plot:

- plots the individual data points;
- dots aren't connected with a line;
- is great for plotting two numerical features (example: correlation analysis).

### Drawing multiple plots on one figure

This can be done by first instantiating the figure and two axis and the using each axis to plot the data. Example taken from [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots).

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.suptitle('Sharing Y axis')

ax1.plot(x, y)
ax2.scatter(x, y)

plt.show()
```

![w01_multiplot.png](./assets/w01_multiplot.png "w01_multiplot.png")

### The logarithmic scale

Sometimes the correlation analysis between two variables can be done easier when one or all of them is plotted on a logarithmic scale. This is because we would reduce the difference between large values as this scale "squashes" large numbers:

![w01_logscale.png](./assets/w01_logscale.png "w01_logscale.png")

In `matplotlib` we can use the [plt.xscale](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xscale.html) function to change the scaling of an axis using `plt` or [ax.set_xscale](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html#matplotlib.axes.Axes.set_xscale) to set the scale of an axis of a subplot.

### Histogram

#### Introduction

The histogram is a plot that's useful to explore **distribution of numeric** data;

Imagine `12` values between `0` and `6`.

![w01_histogram_ex1.png](./assets/w01_histogram_ex1.png "w01_histogram_ex1.png")

To build a histogram for these values, you can divide the line into **equal chunks**, called **bins**. Suppose you go for `3` bins, that each have a width of `2`:

![w01_histogram_ex2.png](./assets/w01_histogram_ex2.png "w01_histogram_ex2.png")

Next, you count how many data points sit inside each bin. There's `4` data points in the first bin, `6` in the second bin and `2` in the third bin:

![w01_histogram_ex3.png](./assets/w01_histogram_ex3.png "w01_histogram_ex3.png")

Finally, you draw a bar for each bin. The height of the bar corresponds to the number of data points that fall in this bin. The result is a histogram, which gives us a nice overview on how the `12` values are **distributed**. Most values are in the middle, but there are more values below `2` than there are above `4`:

![w01_histogram_ex4.png](./assets/w01_histogram_ex4.png "w01_histogram_ex4.png")

#### In `matplotlib`

In `matplotlib` we can use the `.hist` function. In its documentation there're a bunch of arguments you can specify, but the first two are the most used ones:

- `x` should be a list of values you want to build a histogram for;
- `bins` is the number of bins the data should be divided into. Based on this number, `.hist` will automatically find appropriate boundaries for all bins, and calculate how may values are in each one. If you don't specify the bins argument, it will by `10` by default.

![w01_histogram_matplotlib.png](./assets/w01_histogram_matplotlib.png "w01_histogram_matplotlib.png")

The number of bins is important in the following way:

- too few bins will oversimplify reality and won't show you the details;
- too many bins will overcomplicate reality and won't show the bigger picture.

Experimenting with different numbers and/or creating multiple plots on the same canvas can alleviate that.

Here's the code that generated the above example:

```python
import matplotlib.pyplot as plt
xs = [0, 0.6, 1.4, 1.6, 2.2, 2.5, 2.6, 3.2, 3.5, 3.9, 4.2, 6]
plt.hist(xs, bins=3)
plt.show()
```

and the result of running it:

![w01_histogram_matplotlib_code.png](./assets/w01_histogram_matplotlib_code.png "w01_histogram_matplotlib_code.png")

#### Use cases

Histograms are really useful to give a bigger picture. As an example, have a look at this so-called **population pyramid**. The age distribution is shown, for both males and females, in the European Union.

![w01_population_pyramid.png](./assets/w01_population_pyramid.png "w01_population_pyramid.png")

Notice that the histograms are flipped 90 degrees; the bins are horizontal now. The bins are largest for the ages `40` to `44`, where there are `20` million males and `20` million females. They are the so called baby boomers. These are figures of the year `2010`. What do you think will have changed in `2050`?

Let's have a look.

![w01_population_pyramid_full.png](./assets/w01_population_pyramid_full.png "w01_population_pyramid_full.png")

The distribution is flatter, and the baby boom generation has gotten older. **With the blink of an eye, you can easily see how demographics will be changing over time.** That's the true power of histograms at work here!

### Checkpoint

<details>

<summary>
You want to visually assess if the grades on your exam follow a particular distribution. Which plot do you use?

```text
A. Line plot.
B. Scatter plot.
C. Histogram.
```

</summary>

Answer: C.

</details>

<details>

<summary>
You want to visually assess if longer answers on exam questions lead to higher grades. Which plot do you use?

```text
A. Line plot.
B. Scatter plot.
C. Histogram.
```

</summary>

Answer: B.

</details>

### Customization

Creating a plot is one thing. Making the correct plot, that makes the message very clear - that's the real challenge.

For each visualization, you have many options:

- change colors;
- change shapes;
- change labels;
- change axes, etc., etc.

The choice depends on:

- the data you're plotting;
- the story you want to tell with this data.

Below are outlined best practices when it comes to creating an MVP plot.

If we run the script for creating a line plot, we already get a pretty nice plot:

![w01_plot_basic.png](./assets/w01_plot_basic.png "w01_plot_basic.png")

It shows that the population explosion that's going on will have slowed down by the end of the century.

But some things can be improved:

- **axis labels**;
- **title**;
- **ticks**.

#### Axis labels

The first thing you always need to do is label your axes. We can do this by using the `xlabel` and `ylabel` functions. As inputs, we pass strings that should be placed alongside the axes.

![w01_plot_axis_labels.png](./assets/w01_plot_axis_labels.png "w01_plot_axis_labels.png")

#### Title

We're also going to add a title to our plot, with the `title` function. We pass the actual title, `'World Population Projections'`, as an argument:

![w01_plot_title.png](./assets/w01_plot_title.png "w01_plot_title.png")

#### Ticks

Using `xlabel`, `ylabel` and `title`, we can give the reader more information about the data on the plot: now they can at least tell what the plot is about.

To put the population growth in perspective, the y-axis should start from `0`. This can be achieved by using the `yticks` function. The first input is a list, in this example with the numbers `0` up to `10`, with intervals of `2`:

![w01_plot_ticks.png](./assets/w01_plot_ticks.png "w01_plot_ticks.png")

Notice how the curve shifts up. Now it's clear that already in `1950`, there were already about `2.5` billion people on this planet.

Next, to make it clear we're talking about billions, we can add a second argument to the `yticks` function, which is a list with the display names of the ticks. This list should have the same length as the first list.

![w01_plot_tick_labels.png](./assets/w01_plot_tick_labels.png "w01_plot_tick_labels.png")

#### Adding more data

Finally, let's add some more historical data to accentuate the population explosion in the last `60` years. If we run the script once more, three data points are added to the graph, giving a more complete picture.

![w01_plot_more_data.png](./assets/w01_plot_more_data.png "w01_plot_more_data.png")

#### `plt.tight_layout()`

##### Problem

With the default Axes positioning, the axes title, axis labels, or tick labels can sometimes go outside the figure area, and thus get clipped.

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ax = plt.subplots()
example_plot(ax, fontsize=24)
plt.show()
```

![w01_tight_layout_1.png](./assets/w01_tight_layout_1.png "w01_tight_layout_1.png")

##### Solution

To prevent this, the location of Axes needs to be adjusted. `plt.tight_layout()` does this automatically:

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ax = plt.subplots()
example_plot(ax, fontsize=24)
plt.tight_layout()
plt.show()
```

![w01_tight_layout_2.png](./assets/w01_tight_layout_2.png "w01_tight_layout_2.png")

When you have multiple subplots, often you see labels of different Axes overlapping each other:

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.show()
```

![w01_tight_layout_3.png](./assets/w01_tight_layout_3.png "w01_tight_layout_3.png")

`plt.tight_layout()` will also adjust spacing between subplots to minimize the overlaps:

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.tight_layout()
plt.show()
```

![w01_tight_layout_4.png](./assets/w01_tight_layout_4.png "w01_tight_layout_4.png")

## Seaborn

Seaborn builds on Matplotlib, providing a high-level interface for statistical graphics with attractive defaults, themes, and tight integration with Pandas DataFrames.

In the course you can use both `matplotlib` and `seaborn`! **If you want to only work with `seaborn`, this is completely fine!** Use a plotting library of your choice for any plotting exercises in the course.

I suggest you open up a blank Jupyter Notebook so that you can more easily visualize what is plotted.

### 0. Importing

By convention `seaborn` is imported with the alias `sns`:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### 1. Figure, Axes, and Styling

Set up figures/axes (Matplotlib) and styles/themes (Seaborn).

Use the documentation (Matplotlib: <https://matplotlib.org/stable/api/index.html>, Seaborn: <https://seaborn.pydata.org/api.html>) to fill in the missing parts (marked with `???`).

| Function/Method | Library | Problem Solved | Example |
|-----------------|---------|----------------|---------|
| `plt.figure(figsize=(w, h))` | Matplotlib | Creates a figure. | `plt.figure(figsize=(8, 6))`  # Figure 8x6 inches |
| `???` | Seaborn | Sets plot theme. | `???`  # Applies 'darkgrid' theme to all plots |
| `fig, ax = plt.subplots()` | Matplotlib | Creates figure and axes. | `fig, ax = plt.subplots(); ax.plot(...); plt.show()`  # Single axes for plotting |
| `sns.set_context('context')` | Seaborn | Adjusts plot scale. | `sns.set_context('paper')`  # Scales elements for paper-sized plots |

<details>
<summary>Reveal answer</summary>

| Function/Method | Library | Problem Solved | Example |
|-----------------|---------|----------------|---------|
| `plt.figure(figsize=(w, h))` | Matplotlib | Creates a figure. | `plt.figure(figsize=(8, 6))`  # Figure 8x6 inches |
| `sns.set_theme(style='style')` | Seaborn | Sets plot theme. | `sns.set_theme(style='darkgrid')`  # Applies 'darkgrid' theme to all plots |
| `fig, ax = plt.subplots()` | Matplotlib | Creates figure and axes. | `fig, ax = plt.subplots(); ax.plot(...); plt.show()`  # Single axes for plotting |
| `sns.set_context('context')` | Seaborn | Adjusts plot scale. | `sns.set_context('paper')`  # Scales elements for paper-sized plots |

</details>

### 2. Basic Plots: Line and Scatter

Compare line and scatter plots in both libraries. Seaborn adds statistical features and better defaults.

Use the documentation (Matplotlib: <https://matplotlib.org/stable/api/index.html>, Seaborn: <https://seaborn.pydata.org/api.html>) to fill in the missing parts (marked with `???`).

```python
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
```

| Plot Type | Matplotlib Example | Seaborn Example |
|-----------|--------------------|-----------------|
| Line Plot | `???; plt.show()`  # Basic sine wave line plot | `???; plt.show()`  # Sine wave with confidence intervals (if data has replicates) |
| Scatter Plot | `???; plt.show()`  # Points for sine wave | `???; plt.show()`  # Scatter with theme; can add hue/size for categories |

<details>
<summary>Reveal answer</summary>

| Plot Type | Matplotlib Example | Seaborn Example |
|-----------|--------------------|-----------------|
| Line Plot | `plt.plot(x, y); plt.show()`  # Basic sine wave line plot | `sns.lineplot(x=x, y=y); plt.show()`  # Sine wave with confidence intervals (if data has replicates) |
| Scatter Plot | `plt.scatter(x, y); plt.show()`  # Points for sine wave | `sns.scatterplot(x=x, y=y); plt.show()`  # Scatter with theme; can add hue/size for categories |

</details>

### 3. Categorical Plots: Bar and Box

Compare bar plots and add Seaborn's boxplot (statistical). Seaborn excels at categorical data.

Use the documentation (Matplotlib: <https://matplotlib.org/stable/api/index.html>, Seaborn: <https://seaborn.pydata.org/api.html>) to fill in the missing parts (marked with `???`).

```python
df_brics = pd.DataFrame({
    'country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'area': [8.516, 17.10, 3.286, 9.597, 1.221]
})
```

| Plot Type | Matplotlib Example | Seaborn Example |
|-----------|--------------------|-----------------|
| Bar Plot | `???; plt.show()`  # Vertical bars for area by country | `???; plt.show()`  # Bar plot with error bars and theme |
| Box Plot | `???; plt.show()`  # Basic boxplot of area values | `???; plt.show()`  # Boxplot per country with outliers |

<details>
<summary>Reveal answer</summary>

| Plot Type | Matplotlib Example | Seaborn Example |
|-----------|--------------------|-----------------|
| Bar Plot | `plt.bar(df_brics['country'], df_brics['area']); plt.show()`  # Vertical bars for area by country | `sns.barplot(x='country', y='area', data=df_brics); plt.show()`  # Bar plot with error bars and theme |
| Box Plot | `plt.boxplot(df_brics['area']); plt.show()`  # Basic boxplot of area values | `sns.boxplot(x='country', y='area', data=df_brics); plt.show()`  # Boxplot per country with outliers |

</details>

### 4. Distribution Plots: Histogram and KDE

Compare histograms; Seaborn adds KDE (kernel density estimate) for smoother distributions.

Use the documentation (Matplotlib: <https://matplotlib.org/stable/api/index.html>, Seaborn: <https://seaborn.pydata.org/api.html>) to fill in the missing parts (marked with `???`).

```python
import numpy as np
data = np.random.randn(1000)
```

| Plot Type | Matplotlib Example | Seaborn Example |
|-----------|--------------------|-----------------|
| Histogram | `???; plt.show()`  # Histogram with 30 bins of normal distribution | `???; plt.show()`  # Histogram with 30 bins with optional KDE overlay |
| KDE Plot | `from scipy.stats import gaussian_kde; kde = gaussian_kde(data); x = np.linspace(-3, 3, 100); plt.plot(x, kde(x)); plt.show()`  # Manual KDE curve | `???; plt.show()`  # Smooth KDE plot of distribution |

<details>
<summary>Reveal answer</summary>

| Plot Type | Matplotlib Example | Seaborn Example |
|-----------|--------------------|-----------------|
| Histogram | `plt.hist(data, bins=30); plt.show()`  # Histogram of normal distribution | `sns.histplot(data, bins=30); plt.show()`  # Histogram with optional KDE overlay |
| KDE Plot | `from scipy.stats import gaussian_kde; kde = gaussian_kde(data); x = np.linspace(-3, 3, 100); plt.plot(x, kde(x)); plt.show()`  # Manual KDE curve | `sns.kdeplot(data); plt.show()`  # Smooth KDE plot of distribution |

</details>

### 5. Advanced Statistical Plots

Seaborn excels at creating statistical visualizations like pairplots, heatmaps, and violin plots with minimal code, leveraging Pandas DataFrames for ease.

Matplotlib can achieve similar plots but requires more manual setup.

Use the documentation (Matplotlib: <https://matplotlib.org/stable/api/index.html>, Seaborn: <https://seaborn.pydata.org/api.html>) to answer the questions below.

```python
df_brics = pd.DataFrame({
    'country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'area': [8.516, 17.10, 3.286, 9.597, 1.221],
    'population': [200.4, 143.5, 1252, 1357, 52.98]
})

corr = df_brics[['area', 'population']].corr() # correlation matrix for heatmap we'll need shortly
df_titanic= sns.load_dataset("titanic")
```

#### Pair Plot

In Matplotlib we'd need to manually create a grid of subplots to show scatter plots and histograms for pairwise relationships:

```python
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
# Histogram: area vs area (diagonal)
axs[0, 0].hist(df_brics['area'], bins=5)
axs[0, 0].set_xlabel('Area')
axs[0, 0].set_ylabel('Count')
# Scatter: area vs population
axs[0, 1].scatter(df_brics['area'], df_brics['population'])
axs[0, 1].set_xlabel('Area')
axs[0, 1].set_ylabel('Population')
# Scatter: population vs area
axs[1, 0].scatter(df_brics['population'], df_brics['area'])
axs[1, 0].set_xlabel('Population')
axs[1, 0].set_ylabel('Area')
# Histogram: population (diagonal)
axs[1, 1].hist(df_brics['population'], bins=5)
axs[1, 1].set_xlabel('Population')
axs[1, 1].set_ylabel('Count')
plt.tight_layout()
plt.show()
```

![w01_mpl_pair_plot.png](./assets/w01_mpl_pair_plot.png "w01_mpl_pair_plot.png")

<details>
<summary>How can this be achieved via Seaborn?</summary>

Use `pairplot` for a quick, automatic grid of pairwise relationships with histograms on the diagonal:

```python
sns.pairplot(df_brics[['area', 'population']])
plt.show()
```

![w01_sns_pair_plot.png](./assets/w01_sns_pair_plot.png "w01_sns_pair_plot.png")

</details>

#### Heatmap

In Matplotlib we'd use `imshow` to plot a correlation matrix, requiring manual axis labeling.

```python
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.xticks([0, 1], ['area', 'population'])
plt.yticks([0, 1], ['area', 'population'])
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
```

![w01_mpl_heatmap.png](./assets/w01_mpl_heatmap.png "w01_mpl_heatmap.png")

<details>
<summary>How can this be achieved via Seaborn?</summary>

Use `heatmap` for an annotated, styled correlation matrix:

```python
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
```

![w01_sns_heatmap.png](./assets/w01_sns_heatmap.png "w01_sns_heatmap.png")

</details>

#### Violin Plot

In Matplotlib we'd use `violinplot`, but would have to construct groups manually:

```python
classes = sorted(df_titanic['class'].unique())
data = [df_titanic.loc[df_titanic['class'] == c, 'age'].dropna() for c in classes]
fig, ax = plt.subplots(figsize=(8, 5))
parts = ax.violinplot(data, vert=False, showmeans=False, showmedians=True)
ax.set_yticks(range(1, len(classes) + 1))
ax.set_yticklabels(classes)
ax.set_xlabel("age")
ax.set_ylabel("class")
plt.tight_layout()
plt.show()
```

![w01_mpl_violinplot.png](./assets/w01_mpl_violinplot.png "w01_mpl_violinplot.png")

<details>
<summary>How can this be achieved via Seaborn?</summary>

Use `violinplot` for categorical distributions, automatically handling groups.

```python
sns.violinplot(data=df_titanic, x="age", y="class")
plt.show()
```

![w01_sns_violinplot.png](./assets/w01_sns_violinplot.png "w01_sns_violinplot.png")

</details>

### 6. Customization and Subplots

Customize and create subplots. Remember that Seaborn uses Matplotlib's infrastructure.

Use the documentation (Matplotlib: <https://matplotlib.org/stable/api/index.html>, Seaborn: <https://seaborn.pydata.org/api.html>) to fill in the missing parts (marked with `???`).

```python
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
```

| Function/Method | Library | Problem Solved | Example |
|-----------------|---------|----------------|---------|
| `???` | Matplotlib | Sets title. | `???`  # Adds title 'Sine Wave' |
| `???` | Seaborn | Sets color palette. | `???`  # Applies 'husl' color palette to plots |
| `fig, axes = plt.subplots(nrows, ncols)` | Matplotlib | Creates subplots. | `fig, axes = plt.subplots(1, 2); sns.lineplot(x=x, y=y, ax=axes[0]); plt.show()`  # Two subplots, first with Seaborn lineplot |

<details>
<summary>Reveal answer</summary>

| Function/Method | Library | Problem Solved | Example |
|-----------------|---------|----------------|---------|
| `plt.title(label)` | Matplotlib | Sets title. | `plt.title('Sine Wave')`  # Adds title 'Sine Wave' |
| `sns.set_palette('palette')` | Seaborn | Sets color palette. | `sns.set_palette('husl')`  # Applies 'husl' color palette to plots |
| `fig, axes = plt.subplots(nrows, ncols)` | Matplotlib (used by Seaborn) | Creates subplots. | `fig, axes = plt.subplots(1, 2); sns.lineplot(x=x, y=y, ax=axes[0]); plt.show()`  # Two subplots, first with Seaborn lineplot |

</details>

### 7. Saving and Displaying Plots

Save/display plots (shared by both libraries).

Use the documentation (Matplotlib: <https://matplotlib.org/stable/api/index.html>, Seaborn: <https://seaborn.pydata.org/api.html>) to fill in the missing parts (marked with `???`).

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
```

| Function/Method | Problem Solved | Example |
|-----------------|----------------|---------|
| `???` | Displays plot. | `???`  # Shows plot in window/notebook |
| `???` | Saves plot. | `???`  # Saves as 'plot.png' |

<details>
<summary>Reveal answer</summary>

| Function/Method | Problem Solved | Example |
|-----------------|----------------|---------|
| `plt.show()` | Displays plot. | `plt.show()`  # Shows plot in window/notebook |
| `plt.savefig(fname)` | Saves plot. | `plt.savefig('plot.png')`  # Saves as 'plot.png' |

</details>

*Remember*: Seaborn plots are Matplotlib objects; customize with Matplotlib functions (e.g., `ax.set_title()`)

# Week 02 - Machine learning with scikit-learn

## What is machine learning?

<details>

<summary>What are some machine learning use cases you've heard of?</summary>

The point here is to give at least **some** examples.

Here are two possibilities out of many more:

- email: spam vs not spam;
- clustering books into different categories/genres based on their content;
  - assigning any new book to one of the existing clusters.

</details>

<details>

<summary>What is machine learning?</summary>

A process whereby computers learn to make decisions from data without being explicitly programmed.

</details>

<details>

<summary>What types of machine learning do you know?</summary>

Machine learning approaches are traditionally divided into three broad categories, which correspond to learning paradigms:

- Supervised learning;
- Unsupervised learning;
- Reinforcement learning.

</details>

<details>

<summary>What is supervised learning?</summary>

Uncovering patterns in labeled data. Here all possible values to be predicted are already known, and a model is built with the aim of accurately predicting those values on new data.

</details>

<details>

<summary>Can you give an example?</summary>

- Given a labelled set of images (cat or dog), output a label of a new image, not present in the labelled set.
- Given the transactional history of a person, output the likelihood of them being able to pay out their loan.

</details>

</details>

<details>

<summary>What are the main methods of supervised learning?</summary>

- Regression.
- Classification.

</details>

<details>

<summary>What is unsupervised learning?</summary>

Uncovering patterns in unlabeled data.

</details>

</details>

<details>

<summary>Can you give some examples?</summary>

- Clustering books into different categories/genres based on their content.
- Grouping customers into categories based on their purchasing behavior without knowing in advance what those categories are:

![w02_clustering01.png](./assets/w02_clustering01.png "w02_clustering01.png")

</details>

<details>

<summary>What are features?</summary>

Measurable characteristics of the examples that our model uses to predict the value of the target variable.

</details>

<details>

<summary>What are observations?</summary>

The individual samples/examples that our model uses.

</details>

<details>

<summary>Do you know any synonyms of the "feature" term?</summary>

feature = characteristic = predictor variable = independent variable

</details>

<details>

<summary>Do you know any synonyms of the "target variable" term?</summary>

target variable = dependent variable = label = response variable

</details>

<details>

<summary>What features could be used to predict the position of a football player?</summary>

`goals_per_game`, `assists_per_game`, `steals_per_game`, `number_of_passes`

We can represent these features along with the target in a 2D table:

- horizontally, we put the observations;
- vertically, we put their features.

Here an example in the world of basketball:

![w02_basketball_example.png](./assets/w02_basketball_example.png "w02_basketball_example.png")

</details>

<details>

<summary>What is classification?</summary>

Classification is used to predict the label, or category, of an observation.

</details>

<details>

<summary>What are some examples of classification?</summary>

- Predict whether a bank transaction is fraudulent or not. As there are two outcomes here - a fraudulent transaction, or non-fraudulent transaction, this is known as **binary classification**.
- Same is true for spam detection in emails.

</details>

<details>

<summary>What is regression?</summary>

Regression is used to predict continuous values.

</details>

<details>

<summary>What are some examples of regression?</summary>

- A model can use features such as the number of bedrooms, and the size of a property, to predict the target variable - the price of that property.
- Predicting the amount of electricity used in a city based on the day of the year and the events, happening in the city.
- Predicting the amount of rain based on pictures of clouds.

</details>

<details>

<summary>Let's say you want to create a model using supervised learning (for ex. to predict the price of a house). What requirements should the data, you want to use to train the model with, conform to?</summary>

It must:

- not have missing values;
- be in a numerical format;
- stored somewhere in a known format (csv files, parquet files, Amazon S3 buckets, etc.).

</details>

<details>

<summary>How can we make sure that our data conforms to those requirements?</summary>

We must look at our data, explore it. In other words, we need to **perform exploratory data analysis (EDA) first**. Various `pandas` methods for descriptive statistics, along with appropriate data visualizations, are useful in this step.

</details>

<details>

<summary>Do you know what `scikit-learn` is?</summary>

It is a Python package for using already implemented machine learning models and helpful functions centered around the process of creating and evaluating such models. Feel free to take a tour in [it's documentation](https://scikit-learn.org/).

Install using `pip install scikit-learn` and import using `import sklearn`.

</details>

<details>

<summary>Have you heard of any supervised machine learning models?</summary>

Here are some:

- K-Nearest Neighbors (KNN);
- Linear regression;
- Logistic regression;
- Support vector machines (SVM);
- Decision tree;
- Random forest;
- XGBoost;
- CatBoost.

Explore more [in scikit-learn's documentation](https://scikit-learn.org/stable/supervised_learning.html).

</details>

## The `scikit-learn` syntax

`scikit-learn` follows the same syntax for all supervised learning models, which makes the workflow repeatable:

```python
# 1. Import a Model class
from sklearn.module import Model

# 2. Instantiate an object from the Model class
model = Model()

# 3. Fit the model object to your data (X is the array of features, and y is the array of target values)
# Notice the casing:
#   - capital letters represent matrices
#   - lowercase letters represent vectors
# During this step most models learn the patterns about the features and the target variable
model.fit(X, y)

# 4. Use the model's "predict" method, passing X_new - new features of observations to get predictions
predictions = model.predict(X_new)
```

For example, if feeding features from six emails to a spam classification model, an array of six values is returned:

- `1` indicates the model predicts that email is spam;
- `0` indicates a prediction of not spam.

```python
print(predictions)
```

```console
array([0, 0, 0, 0, 1, 0])
```

<details>

<summary>What term is used to refer to the data from which a model learns the patterns?</summary>

As the model learns from the data, we call this the ***training* data**.

</details>

## The classification challenge

Let's say we have the following labelled data - what approaches can we use to assign a label for the back point?

![w02_knn_example.png](./assets/w02_knn_example.png "w02_knn_example.png")

<details>

<summary>Reveal answer</summary>

We can use the geometry of the space and look at the labels of the closest points.

</details>

<details>

<summary>Do you know how the model K-Nearest Neighbors (KNN) works?</summary>

It predicts the label of a data point by:

- looking at the `k` closest labeled data points;
- taking a majority vote.

</details>

<details>

<summary>What class would the black point be assigned to if k = 3?</summary>

The red one, since from the closest three points, two of them are from the red class.

![w02_knn_example2.png](./assets/w02_knn_example2.png "w02_knn_example2.png")

</details>

- `K-Nearest Neighbors` is a **non-linear classification and regression model**:
  - it creates a decision boundary between classes (labels)/values. Here's what it looks like on a dataset of customers who churned vs those who did not:

    ![w02_knn_example3.png](./assets/w02_knn_example3.png "w02_knn_example3.png")

- Using `scikit-learn` to fit the classifier variant of KNN follows the standard syntax:

    ```python
    # import the KNeighborsClassifier from the sklearn.neighbors module
    from sklearn.neighbors import KNeighborsClassifier
    
    # split our data into X, a 2D NumPy array of our features, and y, a 1D NumPy array of the target values
    df_churn = pd.read_csv('https://sagemaker-sample-files.s3.amazonaws.com/datasets/tabular/synthetic/churn.csv')
    X = df_churn[['Day Charge', 'Eve Charge']]
    y = df_churn['Churn?']

    # the target is expected to be a single column with the same number of observations as the feature data
    print(X.shape, y.shape)
    ```

    ```console
    (5000, 2) (5000,)
    ```

    We then instantiate the `KNeighborsClassifier`, setting `n_neighbors=15`, and fit it to the labeled data.

    ```python
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X, y)
    ```

- Predicting unlabeled data also follows the standard syntax:

    Let's say we have a set of new observations, `X_new`. Checking the shape of `X_new`, we see it has three rows and two columns, that is, three observations and two features.

    ```python
    X_new = np.array([[56.8, 17.5],
                      [24.4, 24.1],
                      [50.1, 10.9]])
    print(X_new.shape)
    ```

    ```console
    (3, 2)
    ```

    We use the classifier's `predict` method and pass it the unseen data, again, as a 2D NumPy array of features and observations.

    Printing the predictions returns a binary value for each observation or row in `X_new`. It predicts `1`, which corresponds to `'churn'`, for the first observation, and `0`, which corresponds to `'no churn'`, for the second and third observations.

    ```python
    predictions = knn.predict(X_new)
    print(f'{predictions=}') # notice this syntax! It's valid and cool!
    ```

    ```console
    predictions=[1 0 0]
    ```

## Measuring model performance

<details>

<summary>How do we know if the model is making correct predictions?</summary>

We can evaluate its performance on seen and unseen data during training.

</details>

<details>

<summary>What is a metric?</summary>

A number which characterizes the quality of the model - the higher the metric value is, the better.

</details>

<details>

<summary>What metrics could be useful for the task of classification?</summary>

A commonly-used metric is accuracy. Accuracy is the number of correct predictions divided by the total number of observations:

![w02_accuracy_formula.png](./assets/w02_accuracy_formula.png "w02_accuracy_formula.png")

There are other metrics which we'll explore further.

</details>

<details>

<summary>On which data should accuracy be measured - seen during training or unseen during training?</summary>

We could compute accuracy on the data used to fit the classifier, however, as this data was used to train the model, performance will not be indicative of how well it can **generalize to unseen data**, which is what we are interested in!

We can still measure the training accuracy, but only for book-keeping purposes.

We should split the data into a part that is used to train the model and a part that's used to evaluate it.

![w02_train_test.png](./assets/w02_train_test.png "w02_train_test.png")

We fit the classifier using the training set, then we calculate the model's accuracy against the test set's labels.

![w02_training.png](./assets/w02_training.png "w02_training.png")

Here's how we can do this in Python:

```python
# we import the train_test_split function from the sklearn.model_selection module
from sklearn.model_selection import train_test_split

# We call train_test_split, passing our features and targets.
# 
# parameter test_size: We commonly use 20-30% of our data as the test set. By setting the test_size argument to 0.3 we use 30% here.
# parameter random_state: The random_state argument sets a seed for a random number generator that splits the data. Using the same number when repeating this step allows us to reproduce the exact split and our downstream results.
# parameter stratify: It is best practice to ensure our split reflects the proportion of labels in our data. So if churn occurs in 10% of observations, we want 10% of labels in our training and test sets to represent churn. We achieve this by setting stratify equal to y.
# 
# return value: four arrays: the training data, the test data, the training labels, and the test labels. We unpack these into X_train, X_test, y_train, and y_test, respectively.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# We then instantiate a KNN model and fit it to the training data.
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# To check the accuracy, we use the "score" method, passing X_test and y_test.
print(knn.score(X_test, y_test))
```

```console
0.8800599700149925
```

</details>

<details>

<summary>What is the accuracy of any KNN model on training data when k=1?</summary>

Always 100% because the model has seen the data. For every point we're asking the model to return the class of the closest labelled point, but that closest labelled point is the starting point itself (reflection).

</details>

<details>

<summary>We train a model to predict cats. If our labels have a ratio cats/dogs = 9/1, what would be your conclusion about a model that achieves an accuracy of 88%?</summary>

It is low, since even the greedy strategy of always assigning the most common class, would be more accurate than our model - 90%.

The model that implements the greedy strategy is called the **baseline model**. We should always strive to create a model much better than the baseline model.

</details>

## Model complexity (overfitting and underfitting)

Let's discuss how to interpret `k` in the K-Nearest Neighbors model.

We saw that `KNN` creates decision boundaries, which are thresholds for determining what label a model assigns to an observation.

In the image shown below, as **`k` increases**, the decision boundary is less affected by individual observations, reflecting a **simpler model**:

![w02_k_interpretation.png](./assets/w02_k_interpretation.png "w02_k_interpretation.png")

**Simpler models are less able to detect relationships in the dataset, which is known as *underfitting***. In contrast, complex models can be sensitive to noise in the training data, rather than reflecting general trends. This is known as ***overfitting***.

So, for any `KNN` classifier:

- Larger `k` = Less complex model = Can cause underfitting
- Smaller `k` = More complex model = Can cause overfitting

## Hyperparameter optimization (tuning) / Model complexity curve

<details>

<summary>What are hyperparameters?</summary>

The parameters of the models - the ones that are passed during object instantiation.

</details>

<details>

<summary>What are some hyperparameters of the KNN model?</summary>

- `k`;
- Metric to use for distance computation: `manhattan`, `euclidean`;
- See other hyperparameters in the [documentation of KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier).

</details>

We can also interpret `k` using a model complexity curve - the idea is to calculate the accuracy on the training and test sets using incremental `k` values, and plot the results:

1. Create empty dictionaries to store the train and test accuracies, and an array containing a range of `k` values.
2. Use a `for`-loop to go through the neighbors array and, inside the loop, instantiate a KNN model with `n_neighbors` equal to the current iterator
3. Fit to the training data.
4. Calculate training and test set accuracies, storing the results in their respective dictionaries.

After our `for` loop, we can plot the training and test values, including a legend and labels:

```python
plt.figure(figsize=(8, 6))
plt.title('KNN: Varying Number of Neighbors')
plt.plot(neighbors, train_accuracies.values(), label='Training Accuracy')
plt.plot(neighbors, test_accuracies.values(), label='Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
```

![w02_knn_results.png](./assets/w02_knn_results.png "w02_knn_results.png")

<details>

<summary>What can you conclude from the plot?</summary>

We see that as `k` increases beyond `15` we see underfitting where performance plateaus on both test and training sets. The peak test accuracy actually occurs at around `13` neighbors.

</details>

Which of the following situations looks like an example of overfitting?

```text
A. Training accuracy 50%, testing accuracy 50%.
B. Training accuracy 95%, testing accuracy 95%.
C. Training accuracy 95%, testing accuracy 50%.
D. Training accuracy 50%, testing accuracy 95%.
```

<details>

<summary>Reveal answer</summary>

Answer: C.

</details>

## The Model Report

Whenever we're building a model, we're going to have to produce the so-called **Model report**. This is an Excel file, but serves as the basis of our work and tells the story of our journey. It is typically **presented to your clients** and shows what has been tried out, what worked, what didn't and, ultimately, which is the best model for the task.

Here're the guidelines we'll follow:

1. Each row is a hypothesis - a model that was trained and evaluated.
2. The columns are divided into two sets: the first set of columns represent the values of the hyperparameters of the model, the second set: the metrics on the **test** set. Do not use more than `3` metrics.
3. The first row holds the so-called **baseline model**. This model can be only one of two things: if currently there is a deployed model on the client's environment, then it is taken to be the baseline model. Otherwise the baseline model is the greediest statistical model. For example, this is the model that predicts the most common class in classification problems.
4. The columns that show the metrics express both the value of the metric as well as the percentage of change **compared to the *baseline* model** (we're striving for percentage increase, but should report every case).
5. The rightmost column should be titled `Comments` and should hold our interpretation of the model (what do we see as metrics, is it good, is it bad, etc). We may include the so-called `Error Analysis` which details where this model makes mistakes.
6. Above or below the main table there should be a cell that **explicitly** states which is the best model and why.
7. Below the table or in other sheets there should be the following diagrams: `train vs validation metric` (the main metric used) and if the model outputs a loss, we should have a `train vs validation loss` diagram.
8. The table should not be a pandas export - it should be coloured and tell the story of modelling. Bold and/or highlight the entries in which the metric is highest or to which you want to draw attention to.
9. Do not sort the table after completing the experiments - it should be in the order of the created models. This lets you build up on the section `Comments` and easily track the changes made.
10. Do not create a very wide table - it **should be easy to understand which is the best model** in one to two seconds of looking at it. Focus on the user experience.
11. Optionally, you could create an additional sheet for the best model in which you put 4-5 examples of correct and incorrect predictions. This will control the client's expectations.

> **Tip**: Since we're talking about doing a lot of experiments (typically `50` - `200`), you'll find it tedious to use Jupyter notebooks. Instead, create **scripts** and run them **in parallel**. This will speed up modelling speed tremendously!

The end result is a table that is present in most scientific papers. Here are some examples:

- [EXAMS-V: A Multi-Discipline Multilingual Multimodal Exam Benchmark for Evaluating Vision Language Models](https://arxiv.org/pdf/2403.10378)

![w02_ex_table1.png](./assets/w02_ex_table1.png "w02_ex_table1.png")

- [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

![w02_ex_table2.png](./assets/w02_ex_table2.png "w02_ex_table2.png")

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)

![w02_ex_table3.png](./assets/w02_ex_table3.png "w02_ex_table3.png")

# Week 03 - Regression

## Regression problems

<details>

<summary>What values does the target variable have in regression problems?</summary>

Continuous values.

</details>

<details>

<summary>What are some examples of such values?</summary>

Any number: `weight`, `price`, `temperature`, `score`, `country's GDP`, etc.

</details>

Let's say we have a dataset containing women's health data. Our goal is to create a model that predicts blood glucose levels.

<details>

<summary>How can we approach this problem? / What is the first step we do when creating a model?</summary>

Exploratory data analysis! We start creating plots to see the predictive power of the features.

</details>

Let's say that we find a feature $x$ (ex. `insulin` levels) that when plotted against $y$ ($y$ = blood glucose level) produces this scatter plot:

![w03_sample_data.png](./assets/w03_sample_data.png "w03_sample_data.png")

<details>

<summary>What does this tell us about the feature - is it useful to use it when training the model?</summary>

Yep! It is very much positively correlated with the target, thus we could even say, we only need this one feature to create our model!

</details>

<details>

<summary>What would be the geometric equivalent to our model (if we were to draw the model, what would it be)?</summary>

It'd have to be the line of best fit - the one that comes as close as possible to the actual blood glucose levels:

![w03_sample_data_sol.png](./assets/w03_sample_data_sol.png "w03_sample_data_sol.png")

</details>

<details>

<summary>What is linear regression then?</summary>

A **statistical model** that estimates the **relationship** between a scalar response (dependent variable) and one or more explanatory variables (regressor or independent variable).

The method through which that model is created is called **regression analysis**.

> **Note**: At least **some** relationship **must exist** in order for us to conclude that we can use a regression model for this task.

</details>

## Regression mechanics

Let's see how we can obtain this line.

We'll need to small recap of Linear Algebra to see how things are related.

At the heart of Linear Algebra is the **linear transformation**.

<details>

<summary>What are transformations in the context of Linear Algebra?</summary>

Any function that takes a **single vector as input** and **outputs a single vector**.

The word `transformation` suggests that we think in terms of **movement** in $n$-d space.

![w03_transformation.gif](./assets/w03_transformation.gif "w03_transformation.gif")

</details>

<details>

<summary>What are linear transformations?</summary>

Transformations that keep grid lines **parallel** and **evenly spaced**.

![w03_linear_transformation.gif](./assets/w03_linear_transformation.gif "w03_linear_transformation.gif")

</details>

<details>

<summary>What is a matrix?</summary>

A numerical description of a linear transformation.

![w03_matrix.gif](./assets/w03_matrix.gif "w03_matrix.gif")

</details>

<details>

<summary>What does it mean visually for a matrix to have linearly dependent columns?</summary>

The vectors on which the basis vectors land on are scaled versions of each other:

![w03_matrix_lin_dep.gif](./assets/w03_matrix_lin_dep.gif "w03_matrix_lin_dep.gif")

</details>

<details>

<summary>What is the visual interpretation of the determinant of a matrix?</summary>

![w03_matrix_det.gif](./assets/w03_matrix_det.gif "w03_matrix_det.gif")

</details>

<details>

<summary>What is the definition of the determinant of a matrix then?</summary>

The factor by which a linear transformation changes areas.

To recap the formulas used to compute the determinants, we can refer to the [Wikipedia page](https://en.wikipedia.org/wiki/Determinant).

</details>

<details>

<summary>When is the determinant zero?</summary>

When the columns of the matrix are linearly dependent.

![w03_matrix_lower_dim.gif](./assets/w03_matrix_lower_dim.gif "w03_matrix_lower_dim.gif")

</details>

<details>

<summary>What is the inverse of a matrix?</summary>

In general, $A^{-1}$ is the unique transformation with the property that if you apply the transformation $A$, and follow it with the transformation $A$ inverse, you end up back where you started.

![w03_sheer_inverse.gif](./assets/w03_sheer_inverse.gif "w03_sheer_inverse.gif")

</details>

<details>

<summary>Why is the inverse helpful?</summary>

It allows us to solve linear systems of equations.

![w03_usecase_inverse.png](./assets/w03_usecase_inverse.png "w03_usecase_inverse.png")

</details>

<details>

<summary>Which matrices are not invertible?</summary>

The ones that have a determinant of $0$.

</details>

Let's say that we have $Ax = b$.

<details>

<summary>If A is not invertible, does that mean there is no solution?</summary>

A solution exists if and only if $b$ lies in the lower dimensional space or is the $0$ vector:

![w03_sol_when_no_inverse.png](./assets/w03_sol_when_no_inverse.png "w03_sol_when_no_inverse.png")

</details>

<details>

<summary>What is the "rank" of a matrix?</summary>

- The number of dimensions in the output of a transformation.
- The number of linearly independent columns.

</details>

What is the rank of this matrix?

$$
\begin{bmatrix}
1 & -2 & 4 \\
-2 & 4 & -8 \\
5 & -10 & 20
\end{bmatrix}
$$

<details>

<summary>Reveal answer</summary>

The columns of this matrix can be expressed as
\[
\begin{bmatrix}
1 \\ -2 \\ 5
\end{bmatrix}
= -\tfrac{1}{2}
\begin{bmatrix}
-2 \\ 4 \\ -10
\end{bmatrix}
= \tfrac{1}{4}
\begin{bmatrix}
4 \\ -8 \\ 20
\end{bmatrix}.
\]
Since they are all linearly dependent and form a line, the rank is $1$.

</details>

What's the dot product of $\left[\begin{smallmatrix}4\\6\end{smallmatrix}\right]$ and $\left[\begin{smallmatrix}-3\\2\end{smallmatrix}\right]$?

<details>

<summary>Reveal answer</summary>

$4(-3) + 6(2) = -12 + 12 = 0$

</details>

<details>

<summary>What does this mean for these vectors?</summary>

They are perpendicular to each other.

![w03_zero_dot.png](./assets/w03_zero_dot.png "w03_zero_dot.png")

</details>

<details>

<summary>What is the geometric interpretation of the dot product?</summary>

We project one onto the other and take multiply their lengths:

![w03_dot_product_viz.gif](./assets/w03_dot_product_viz.gif "w03_dot_product_viz.gif")

</details>

<details>

<summary>What is relationship between dot products and matrix-vector multiplication?</summary>

The dual of a vector is the linear transformation it encodes, and the dual of a linear transformation from some space to one dimension is a certain vector in that space.

$$a \cdot b = a^Tb$$

![w03_duality.png](./assets/w03_duality.png "w03_duality.png")

</details>

Ok, awesome, let's now go back to our example:

![w03_sample_data_sol.png](./assets/w03_sample_data_sol.png "w03_sample_data_sol.png")

<details>

<summary>How can we write the above in the context of equations?</summary>

If we label the points as follows:

![w03_sample_data_sol_lbl.png](./assets/w03_sample_data_sol_lbl.png "w03_sample_data_sol_lbl.png")

Then we can produce three equations:

$$
\begin{align*}
y_1 &= \beta_0 + \beta_1x_1 \\
y_2 &= \beta_0 + \beta_1x_2 \\
y_3 &= \beta_0 + \beta_1x_3
\end{align*}
$$

</details>

<details>

<summary>What can go wrong in the above equations?</summary>

We have three equations, but only $2$ unknowns. The system might be inconsistent, i.e. there doesn't exist a solution.

There might not be any single line that can pass through our datapoints (which as we can see is indeed the case).

</details>

<details>

<summary>What does this mean geometrically - how can we visualize the problem?</summary>

The columns of this matrix can be expressed as

\[
\begin{bmatrix}
y_1 \\ y_2 \\ y_3
\end{bmatrix}
= \beta_0
\begin{bmatrix}
1 \\ 1 \\ 1
\end{bmatrix} +
\beta_1
\begin{bmatrix}
x_1 \\ x_2 \\ x_3
\end{bmatrix}.
\]

And we'll see that actually **$y$ cannot be reached via a linear combination of the $1$s and the $x$s**:

![w03_sample_data_sol_prb.png](./assets/w03_sample_data_sol_prb.png "w03_sample_data_sol_prb.png")

If there was a solution, then $y$ would lie in the plane.

</details>

<details>

<summary>Hmm - ok, this is a bummer - what can we do?</summary>

The strategy of least squares is:

1. Project $y$ onto the plane.
2. Choose $\beta$ values which can get us to that projection.

![w03_sample_data_strat.png](./assets/w03_sample_data_strat.png "w03_sample_data_strat.png")

</details>

But, how do we choose this $\hat{y}$?

<details>

<summary>Reveal answer</summary>

Well, we want to get the best model, right?

So the question is "Which choice of $\hat{y}$ comes as close as possible to the actual $y$?".

It must be where the **rejection vector** ($y - \hat{y}$) is perpendicular to the plane. Any other choice would be farther away.

Minimizing this distance is why the method is referred to as *least squares*. The loss function we'll see shortly - sum of squared residuals is the squared length of this rejection vector.

</details>

Ok, so now we have $\hat{y}$ (orthogonal projection of $y$) - how do we compute the values for the $\beta$s?

<details>

<summary>Reveal answer</summary>

Since the projection and rejection are orthogonal to the plane, they are orthogonal to the vectors with $1$ and $x$:

$$(y - \hat{y}) \cdot 1 = 0$$

and

$$(y - \hat{y}) \cdot x = 0$$

Let's rearrange to get positive signs:

$$1 \cdot \hat{y} = 1 \cdot y$$

$$x \cdot \hat{y} = x \cdot y$$

We know what $\hat{y}$ is:

$$1 \cdot (\beta_01 + \beta_1x)= 1 \cdot y$$

$$x \cdot (\beta_01 + \beta_1x)= x \cdot y$$

Removing the parenthesis:

$$\beta_01 \cdot 1 + \beta_11 \cdot x = 1 \cdot y$$

$$\beta_0x \cdot 1 + \beta_1x \cdot x = x \cdot y$$

And we get to a matrix format:

\[
\begin{bmatrix}
1 \cdot 1 & 1 \cdot x \\ x \cdot 1 & x \cdot x
\end{bmatrix}
\begin{bmatrix}
\beta_0 \\ \beta_1
\end{bmatrix}
= \begin{bmatrix}
1 \cdot y \\ x \cdot y
\end{bmatrix}
\]

Let's use the dual property of vectors:

\[
\begin{bmatrix}
1^T 1 & 1^T x \\ x^T 1 & x^T x
\end{bmatrix}
\begin{bmatrix}
\beta_0 \\ \beta_1
\end{bmatrix}
= \begin{bmatrix}
1^T y \\ x^T y
\end{bmatrix}
\]

And we can further decompose the left-most matrix:

\[
\begin{bmatrix}
1^T \\ x^T
\end{bmatrix}
\begin{bmatrix}
1 & x
\end{bmatrix}
\begin{bmatrix}
\beta_0 \\ \beta_1
\end{bmatrix}
= \begin{bmatrix}
1^T y \\ x^T y
\end{bmatrix}
\]

And take out the repeating $y$:

\[
\begin{bmatrix}
1^T \\ x^T
\end{bmatrix}
\begin{bmatrix}
1 & x
\end{bmatrix}
\begin{bmatrix}
\beta_0 \\ \beta_1
\end{bmatrix}
= \begin{bmatrix}
1^T \\ x^T
\end{bmatrix} y
\]

Now, if we put all the features into their own matrix, we'll get the famous **normal equation** associated with $X\beta = y$:

\[
X=
\begin{bmatrix}
1 & x
\end{bmatrix}
\]

\[
X^T X
\begin{bmatrix}
\beta_0 \\ \beta_1
\end{bmatrix}
= X^T y
\]

</details>

<details>

<summary>What would be the formula for our model's coefficients?</summary>

We'd have to invert:

\[
\begin{bmatrix}
\beta_0 \\ \beta_1
\end{bmatrix}
= (X^T X)^{-1} X^T y
\]

</details>

<details>

<summary>What would be the formula for the model's predictions?</summary>

We'll use the original formula we got:

$$\hat{y} = \beta_01 + \beta_1x$$

Convert it to a matrix form via the dual property:

\[
\hat{y}
= \begin{bmatrix}
1 & x
\end{bmatrix}
\begin{bmatrix}
\beta_0 \\ \beta_1
\end{bmatrix}
\]

And then substitute with what we just found, firstly the capital $X$:

\[
\hat{y}
= X
\begin{bmatrix}
\beta_0 \\ \beta_1
\end{bmatrix}
\]

And then the formula for the coefficients:

$$\hat{y} = X(X^T X)^{-1} X^T y$$

In addition, $X(X^T X)^{-1} X^T$ is the projection matrix.

</details>

Ok, great! So, everything is nice and easy, just solve the equation and we're good to go, right?

<details>

<summary>Right?</summary>

We'll, we have a small problem here / corner case. It is due to the assumption we're making about the matrix with the features $X$.

The above formula works only if $X^TX$ is **invertible**, i.e. has full rank / contains only independent columns.

</details>

<details>

<summary>Hmm - ok, but what if it's not?</summary>

We have several options here, the first of which we should apply in every situation:

1. Remove the collinear features. This would allows us to invert $X^TX$.
2. Compute the pseudoinverse of $X$ and use it instead.

I'd recommend to always be wary of the features we put into the model and never put collinear ones - that would mean we have redundant information which is irrelevant.

If you're curious to learn how we can compute and use the pseudoinverse of $X$, check out [this Wikipedia article](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) and [this phenomenal lecture](https://www.youtube.com/watch?v=ZUU57Q3CFOU).

</details>

## Modelling via regression

Here are the actual first five rows:

|  idx | pregnancies | glucose | diastolic | triceps | insulin | bmi  | dpf   | age | diabetes |
| ---: | ----------- | ------- | --------- | ------- | ------- | ---- | ----- | --- | -------- |
|    0 | 6           | 148     | 72        | 35      | 0       | 33.6 | 0.627 | 50  | 1        |
|    1 | 1           | 85      | 66        | 29      | 0       | 26.6 | 0.351 | 31  | 0        |
|    2 | 8           | 183     | 64        | 0       | 0       | 23.3 | 0.672 | 32  | 1        |
|    3 | 1           | 89      | 66        | 23      | 94      | 28.1 | 0.167 | 21  | 0        |
|    4 | 0           | 137     | 40        | 35      | 168     | 43.1 | 2.288 | 33  | 1        |

<details>

<summary>What is simple linear regression?</summary>

Using one feature to predict the target:

$$y = ax + b$$

</details>

<details>

<summary>What would be an example of that type of regression with the given dataset?</summary>

Using (**only** the) `insulin` levels to predict the blood glucose levels.

</details>

<details>

<summary>What is multiple linear regression?</summary>

Using at least two features to predict the target:

$$y = a_1x_1 + a_2x_2 + a_3x_3 + \dots + a_nx_n + b$$

</details>

<details>

<summary>What would be an example of that type of regression with the given dataset?</summary>

Using `insulin` levels and the `diastolic` pressure to predict the blood glucose levels.

</details>

We need to decide which feature(s) to use.

<details>

<summary>How can we do it?</summary>

Two options:

- if we have experience in the field: we create the data audit file and based on it create **multiple hypothesis**. We then try them out and pick the features that lead to the model scoring highest on our metrics.
- if we do not have experience in the field: we create the data audit file and discuss it with domain experts (medical personnel, consultants with medical knowledge, clients). They'd guide us in what features make sense and which would end-up introducing more noise.

</details>

Let's say that we talk with internal consultants and they advise us to check whether there's any relationship between between blood glucose levels and body mass index. We plot them using a scatter plot:

![w03_bmi_bg_plot.png](./assets/w03_bmi_bg_plot.png "w03_bmi_bg_plot.png")

We can see that, generally, as body mass index increases, blood glucose levels also tend to increase. This is great - we can use this feature to create our model.

### Data Preparation

To do simple linear regression we slice out the column `bmi` of `X` and use the `[[]]` syntax so that the result is a dataframe (i.e. two-dimensional array) which `scikit-learn` requires when fitting models.

```python
X_bmi = X[['bmi']]
print(X_bmi.shape, y.shape)
```

```console
(752, 1) (752,)
```

### Modelling

Now we're going to fit a regression model to our data.

We're going to use the model `LinearRegression`. It fits a straight line through our data.

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_bmi, y)

predictions = reg.predict(X_bmi)
```

## Model evaluation

### Visually

The most basic form of evaluation is the visual one:

```python
plt.scatter(X_bmi, y) # plot the true target values
plt.plot(X_bmi, predictions) # plot the predicted target values
plt.ylabel('Blood Glucose (mg/dl)')
plt.xlabel('Body Mass Index')
plt.show()
```

![w03_linear_reg_predictions_plot.png](./assets/w03_linear_reg_predictions_plot.png "w03_linear_reg_predictions_plot.png")

The black line represents the linear regression model's fit of blood glucose values against body mass index, which appears to have a weak-to-moderate positive correlation.

<details>

<summary>What is the baseline model in regression tasks?</summary>

The model that always predicts the mean target value in the training data.

</details>

### Using a metric

The default metric for linear regression is $R^2$.

It represents the proportion of variance (of $y$) that has been explained by the independent variables in the model.

If $\hat{y}_i$ is the predicted value of the $i$-th sample and $y_i$ is the corresponding true value for total $n$ samples, the estimated $R^2$ is defined as:

$$R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

where $\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$ and $\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} \epsilon_i^2$.

Best possible score is $1.0$ and **it can be negative** (because the model can be arbitrarily worse).

Here are two plots visualizing high and low R-squared respectively:

![w03_r_sq.png](./assets/w03_r_sq.png "w03_r_sq.png")

To compute $R^2$ in `scikit-learn`, we can call the `.score` method of a linear regression class passing test features and targets. Let's say that for our task we get this:

```python
reg_all.score(X_test, y_test)
```

```console
0.356302876407827
```

<details>

<summary>Is this a good result?</summary>

No. Here the features only explain about 35% of blood glucose level variance.

</details>

<details>

<summary>What would be the score of the baseline model on the training set?</summary>

$0$, because it'll just predict the mean of the target.

$$R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} = 1 - \frac{\sum_{i=1}^{n} (y_i - \bar{y})^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} = 1 - 1 = 0$$

> **Note**: we would still need to evaluate the baseline model on the test set, as the training and test set needn't have the same value for the mean.

</details>

#### Adjusted $R^2$ ($R_{adj}^2$)

Using $R^2$ could have downsides in some situations. In today's tasks you'll investigate what they are and how the extension $R_{adj}^2$ can help.

### Using a loss function

Another way to assess a regression model's performance is to take the mean of the residual sum of squares. This is known as the **mean squared error**, or (`MSE`).

> **Note**: `error` function = `loss` function = `cost` function.

$$MSE = \frac{1}{n} \sum (y_i - \hat{y_i})^2$$

`MSE` is measured in units of our target variable, squared. For example, if a model is predicting a dollar value, `MSE` will be in **dollars squared**.

This is not very easy to interpret. To convert to dollars, we can take the **square root** of `MSE`, known as the **root mean squared error**, or `RMSE`.

$$RMSE = \sqrt{MSE}$$

`RMSE` has the benefit of being in the same unit as the target variable.

To calculate the `RMSE` in `scikit-learn`, we can use the `root_mean_squared_error` function in the `sklearn.metrics` module.

<details>

<summary>What's the main difference then between metrics and loss functions?</summary>

- Metrics are maximized.
- Loss functions are minimized.

</details>

# Week 04 - Regularized Regression, Logistic Regression, Cross Validation

## Logistic Regression

### Binary classification

Let's say we want to predict whether a patient has diabetes. During our data audit we spot the following relationship:

![w04_log_reg_1.png](./assets/w04_log_reg_1.png "w04_log_reg_1.png")

<details>

<summary>What are some examples of a working decision boundary for this task?</summary>

There are many lines that we could use:

![w04_log_reg_3.png](./assets/w04_log_reg_3.png "w04_log_reg_3.png")

</details>

<details>

<summary>What model can we use from the ones we already know about that can solve this task?</summary>

- We could use KNN, but it is prone to overfitting if there are outliers:

![w04_log_reg_2.png](./assets/w04_log_reg_2.png "w04_log_reg_2.png")

- We could also use linear regression, but if there are outliers it is going to get very skewed as well:

If the dataset is perfectly separable, maybe we could use a threshold:

![w04_log_reg_5.png](./assets/w04_log_reg_5.png "w04_log_reg_5.png")

In this case we could say:

If $\beta^{T}x >= 0.5$, predict $y = 1$.
If $\beta^{T}x < 0.5$, predict $y = 0$.

But that solution would break if there's a single outlier:

![w04_log_reg_4.png](./assets/w04_log_reg_4.png "w04_log_reg_4.png")

</details>

<details>

<summary>What other problem does linear regression have in relation to the y-axis?</summary>

It can output any value, actually - much greater than 1 and much less than 0.

</details>

<details>

<summary>So what should be output range of our desired model?</summary>

Ideally, we should get a confidence score / a probability. It spans the range $[0, 1]$ and would be very easy for us to threshold at $0.5$.

</details>

Well, this is that Logistic regression does. Note that despite its name, **it is a *classification* algorithm**, not a regression one.

$$0 <= h_{\beta}(x) <= 1$$

Hmm - so then we only have to define what $h_{\beta}(x)$ is.

<details>

<summary>Do you know what "h" is for logistic regression?</summary>

It's actually pretty close to the one for linear regression.

For linear regression we have: $\beta^{T}x$.
For logistic regression we have: $\sigma(\beta^{T}x)$, where $\sigma$ is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) also known as the logistic function:

$$\sigma(\beta^{T}x) = \frac{1}{1 + e^{-\beta^{T}x}}$$

![w04_log_reg_6.png](./assets/w04_log_reg_6.png "w04_log_reg_6.png")

</details>

<details>

<summary>Seeing the graph, can you explain how the output of this model should be interpreted?</summary>

It is the estimated probability that $y = 1$ on input features $x$:

$$h_{\beta}(x) = P(y = 1 | x; \beta)$$

> **Note 1:** In `sklearn` and in our implementation this logic will be defined in a method `predict_proba`.
> **Note 2:** Typically, we have another method `predict` that does the threshold, typically at $0.5$, i.e. predict $y = 1$ if $h_{\beta}(x) >= 0.5$, and $y = 0$ otherwise.

</details>

<details>

<summary>What's your intuition - can logistic regression produce non-linear decision boundary?</summary>

It can! Unlike in linear regression, we can create a decision boundary in any shape that can be described by a polynomial:

Suppose that $h_{\beta}(x) = \sigma(\beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_3x_{1}^{2} + \beta_4x_{2}^{2})$ and we get the following estimated parameters:

$$\beta_0=-1, \beta_1=0, \beta_2=0, \beta_3=1, \beta_4=1$$

> **Note:** We have correlated features here. This will not stop us from getting our parameters $\beta$, but like we said earlier - it is not encouraged, as they do not add predictive value.

Those coefficients actually describe a non-linear boundary ($x_{1}^{2} + x_{2}^{2} = 1$):

![w04_log_reg_7.png](./assets/w04_log_reg_7.png "w04_log_reg_7.png")

</details>

### Model fitting

Ok, great! We have a powerful model that we can use to create linear and non-linear decision boundaries. The question that remains is: *How do we train it? / How do we obtain the values for the coefficients $\beta$?*. Let's answer it in this section.

<details>

<summary>Last time we showed how we can "train" a linear regression model - what was the goal of this training that allowed us to obtain the best parameters?</summary>

We wanted to reduce the error rate, i.e. minimize the loss of the model which was expressed by the sum of the squared differences between predicted and actual values:

$$J(\beta) = \Sigma(y - \hat{y})^2$$

</details>

The strategy of logistic regression is to **maximize the likelihood of observing our target values**. It postulates that the best model parameters are the ones via which we can do $\beta^{T}x$ and get the target values.

<details>

<summary>Do you know what the formal name of this process that finds such coefficients is?</summary>

[Maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)!

From Wikipedia:

"... a method of estimating the parameters of an assumed probability distribution, given some observed data. This is achieved by maximizing a likelihood function so that, under the assumed statistical model, the observed data is most probable."

</details>

Perfect - let's then see whether we have these ingredients.

<details>

<summary>Do we have observed data?</summary>

Yes, this is any dataset for classification purposes.

</details>

<details>

<summary>Do we know what our model is?</summary>

Yes, it is:

$$h_{\beta}(x) = P(y = 1 | x; \beta) = \sigma(\beta^{T}x) = \frac{1}{1 + e^{-\beta^{T}x}}$$

</details>

<details>

<summary>Do we know the probability distribution of our target variable?</summary>

In the context of a single observation, it follows the [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution) - the distribution for 0s and 1s:

$$y_i \sim Ber(\sigma(\beta^{T}x))$$

</details>

<details>

<summary>Ok, what was P(y = 1)?</summary>

$$P(y = 1) = \sigma(\beta^{T}x) = h_{\beta}(x)$$

</details>

<details>

<summary>What is P(y = 0)?</summary>

$$P(y = 0) = 1 - \sigma(\beta^{T}x) = 1 - h_{\beta}(x)$$

</details>

<details>

<summary>How can we combine the above two equations into one that can show us the probability for either of the two classes for a single observation?</summary>

$$P(y_i) = h_{\beta}(x_i)^{y_i} (1 - h_{\beta}(x_i))^{1 - y_i}$$

</details>

Awesome!

<details>

<summary>What is the general form of the likelihood function?</summary>

$$L(\beta) = P(Y | X; \beta)$$

where $Y$ are our target labels and $X$ are the inputs.

</details>

<details>

<summary>Knowing that Y has a Bernoulli distribution, how can we express the likelihood function?</summary>

Our ($m$) observations are (assumed to be) independent, so this probability is equal to the product of the individual probabilities:

$$L(\beta) = P(Y | X; \beta) = \prod_{i=1}^{m} P(y_i) = \prod_{i=1}^{m} h_{\beta}(x_i)^{y_i} (1 - h_{\beta}(x_i))^{1 - y_i}$$

</details>

<details>

<summary>We're dealing with products of probabilities here - what's a problem that can occur?</summary>

It wouldn't be numerically stable for very small numbers (which we'll get from all those multiplications).

</details>

<details>

<summary>How can we deal with this?</summary>

We can instead maximize the **log likelihood** to get a sum of probabilities:

$$L(\beta) = \sum_{i=1}^{m} \left( \log h_{\beta}(x_i)^{y_i} + \log (1 - h_{\beta}(x_i))^{1 - y_i} \right)$$

After moving the powers to become a multiplier, we get:

$$L(\beta) = \sum_{i=1}^{m} \left( {y_i} \log h_{\beta}(x_i) + (1 - y_i) \log (1 - h_{\beta}(x_i)) \right)$$

</details>

Ok, perfect! So, now we just have to maximize this function and we'll get our parameters $\beta$.

<details>

<summary>But wait a minute - last time we minimized a function (the loss function), why are we maximizing now?</summary>

That's a fair question! We needn't maximize, actually.

</details>

<details>

<summary>How can we still find the best parameters, but without maximizing?</summary>

We can **minimize** the **negative log likelihood** instead!

$$J(\beta) = - \sum_{i=1}^{m} \left( {y_i} \log h_{\beta}(x_i) + (1 - y_i) \log (1 - h_{\beta}(x_i)) \right)$$

Further more we can scale this function, like we did with linear regression and the **mean** squared error to get the average loss per observation:

$$J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} \left( {y_i} \log h_{\beta}(x_i) + (1 - y_i) \log (1 - h_{\beta}(x_i)) \right)$$

</details>

And this right here that we derived is referred to as the **binary cross-entropy** (or **log loss**) is a **loss** function! It is a special case of [the cross-entropy function](https://en.wikipedia.org/wiki/Cross-entropy) ($k = 2$):

$$L_{\log}(Y, \hat{P}) = -\log \operatorname{Pr}(Y|\hat{P}) = - \frac{1}{M} \sum_{i=0}^{M-1} \sum_{k=0}^{K-1} y_{i,k} \log \hat{p}_{i,k}$$

Great! Let's recap:

- Loss function for linear regression: $\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y_i})^2$
- Loss function for logistic regression: $-\frac{1}{m} \sum_{i=1}^{m} \left( {y_i} \log h_{\beta}(x_i) + (1 - y_i) \log (1 - h_{\beta}(x_i)) \right)$

<details>

<summary>What would be our next steps?</summary>

1. Compute the first derivative of the loss.
2. Set it to $0$.
3. Solve.

</details>

<details>

<summary>What do we get from step 1?</summary>

The derivative of the log is:

$$\frac{d}{dx} \log(g(x)) = \frac{g'(x)}{g(x)}$$

so we get:

$$
\frac{\partial J(\beta)}{\partial \beta} = \frac{y}{h_{\beta}(x)} \frac{\partial h_{\beta}(x)}{\partial \beta} + \frac{1 - y}{1 - h_{\beta}(x)} (- \frac{\partial h_{\beta}(x)}{\partial \beta})
$$

> **Note:** Let's ignore the leading negative sign for now.

What's left is to calculate $\frac{\partial h_{\beta}(x)}{\partial \beta} = \frac{\partial \sigma(\beta^{T}x)}{\partial \beta}$:

$$\frac{\partial h_{\beta}(x)}{\partial \beta} = \frac{\partial h_{\beta}(x)}{\partial \beta^{T}x} * \frac{\partial \beta^{T}x}{\partial \beta}$$

We can show that:

$$\frac{\partial \beta^{T}x}{\partial \beta} = x$$

and with the derivative of the sigmoid (which is mathematically convenient):

$$\frac{\partial h_{\beta}(x)}{\partial \beta^{T}x} = h_{\beta}(x) (1 - h_{\beta}(x))$$

we get:

$$\frac{\partial h_{\beta}(x)}{\partial \beta} = h_{\beta}(x) (1 - h_{\beta}(x))x$$

Substituting above:

$$
\frac{\partial J(\beta)}{\partial \beta} = \frac{y}{h_{\beta}(x)} h_{\beta}(x) (1 - h_{\beta}(x))x - \frac{1 - y}{1 - h_{\beta}(x)} h_{\beta}(x) (1 - h_{\beta}(x))x
$$

We can cancel some of the elements:

$$
\frac{\partial J(\beta)}{\partial \beta} = y (1 - h_{\beta}(x))x - (1 - y) h_{\beta}(x) x = yx - yx h_{\beta}(x) - h_{\beta}(x)x + h_{\beta}(x) x y = yx - h_{\beta}(x)x
$$

And we get:

$$
\frac{\partial J(\beta)}{\partial \beta} = (y - h_{\beta}(x))x = (y - \hat{y})x
$$

After taking into account the negative sign we ignored in the beginning, we obtain the final result:

$$
\frac{\partial J(\beta)}{\partial \beta} = (\hat{y} - y)x
$$

</details>

And now we have to set to $0$ and solve.

Except ...

We have a little bit of a problem - there's no closed-form solution, i.e. there is no exact formula that we can obtain. So, we kinda get into the situation with the invertible matrices last time.

<details>

<summary>From the approaches we mentioned last time, which one do you think we'll use?</summary>

We have several options now, but all of them are based on **approximation**:

- fancy algorithms: `sklearn` comes with them - `lbfgs`, `liblinear`, `newton-cg`, `newton-cholesky`, `sag`, `saga`, etc.
- gradient descent.

Let's go with gradient descent.

</details>

<details>

<summary>Do you know what its pseudocode looks like?</summary>

betas = ... initialize with small values around 0... <- this is our starting solution

for i in range(max_iter):
  betas = betas - learning_rate * dJ(beta)

</details>

To monitor the algorithm for converging, you could output the (training) loss at every step.

<details>

<summary>Why - what value would it have?</summary>

It should decrease if we're training properly.

</details>

And that's it! We'll probably not be able to obtain zero loss - again, this is ok so long as our approximation is the best we can do.

### Multiclass classification

<details>

<summary>Do you know how we can solve this?</summary>

We have two main strategies to choose from:

- One-vs-rest logistic regression.
- Multinomial logistic regression.

</details>

<details>

<summary>Do you know how the first one works?</summary>

We create $N$ models, each being an expert for its own class. For a new observation we take the prediction of the most confident model.

</details>

<details>

<summary>Do you know how the second one works?</summary>

We create a model that outputs $N$ probability values. This is more akin to neural networks (for which we'll talking about later in the course), so we'll skip it for now.

</details>

In this week's implementation we'll use the one-vs-rest (`ovr`) strategy.

To convert the raw output values, we'll use the softmax function:

$$softmax(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K$$

### Logistic regression in `scikit-learn`

<details>

<summary>Have you heard about the iris dataset?</summary>

- The [`iris` dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) is a collection of measurements of many [iris plants](https://en.wikipedia.org/wiki/Iris_flower_data_set).
- Three species of iris:
  - *setosa*;
  - *versicolor*;
  - *virginica*.
- Features: petal length, petal width, sepal length, sepal width.

Iris setosa:

![w05_setosa.png](./assets/w05_setosa.png "w05_setosa.png")

Iris versicolor:

![w05_versicolor.png](./assets/w05_versicolor.png "w05_versicolor.png")

Iris virginica:

![w05_virginica.png](./assets/w05_virginica.png "w05_virginica.png")

</details>

In scikit-learn logistic regression is implemented in the [`sklearn.linear_model`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#logisticregression) module:

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
print(f'Possible classes: {set(y)}')
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
```

```console
Possible classes: {0, 1, 2}
array([0, 0])
```

> **Note:** The hyperparameter `C` is the inverse of the regularization strength - larger `C` means less regularization and smaller `C` means more regularization.

```python
clf.predict_proba(X[:2]) # probabilities of each instance belonging to each of the three classes
```

```console
array([[9.81780846e-01, 1.82191393e-02, 1.44184120e-08],
       [9.71698953e-01, 2.83010167e-02, 3.01417036e-08]])
```

```python
clf.predict_proba(X[:2])[:, 2] # probabilities of each instance belonging to the third class only
```

```console
array([1.44184120e-08, 3.01417036e-08])
```

```python
clf.score(X, y) # returns the accuracy
```

```console
0.97
```

## Regularized Regression

### Regularization

<details>

<summary>Have you heard of regularization?</summary>

Regularization is a technique used to avoid overfitting. It can be applied in any task - classification or regression.

![example](https://www.mathworks.com/discovery/overfitting/_jcr_content/mainParsys/image.adapt.full.medium.svg/1718273106637.svg)

Its main idea is to reduce the size / values of model parameters / coefficients as large coefficients lead to overfitting.

Linear regression models minimize a loss function to choose a coefficient - $a$, for each feature, and an intercept - $b$. When we apply regularization to them, we "extend" the loss function, adding one more variable to the sum, that grows in value as coefficients grow.

</details>

### Ridge Regression

The first type of regularized regression that we'll look at is called `Ridge`. With `Ridge`, we use the `Ordinary Least Squares` loss function plus the squared value of each coefficient, multiplied by a constant - `alpha`.

$$J = \sum_{i=1}^n(y_i - \hat{y_i})^2 + \alpha \sum_{i=1}^na_i^2$$

> **Note:** We usually do not apply regularization to the bias term.

So, when minimizing the loss function, models are penalized both *for creating a line that's far from the ideal one* **and** *for coefficients with large positive or negative values*.

When using `Ridge`, we need to choose the `alpha` value in order to fit and predict.

- we can select the `alpha` for which our model performs best;
- picking alpha for `Ridge` is similar to picking `k` in `KNN`;
- multiple experiments with different values required - choose local minimum; hope it is the global one.

`Alpha` controls model complexity. When alpha equals `0`, we are performing `OLS`, where large coefficients are not penalized and overfitting *may* occur. A high alpha means that large coefficients are significantly penalized, which *can* lead to underfitting (we're making our model dumber).

`Scikit-learn` comes with a ready-to-use class for Ridge regression - check it out [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#ridge).

The geometric interpretation is similar to the one we showed last time with the only change being that now the projection is not orthogonal since we're not allowed to use the full scale of the feature vectors:

![w04_geom_interpret_ridge.png](./assets/w04_geom_interpret_ridge.png "w04_geom_interpret_ridge.png")

It can be shown that the best parameters for ridge regression can be obtained if we solve the following equation:

$$
\hat{\beta}_{ridge} = (X^{\top} X + \lambda D)^{-1} X^{\top} y, \quad
D = \text{diag}(0, 1, 1, \dots, 1)
$$

> **Note:** Adding $\lambda D$ **ensures the matrix is invertible** (even if $X^{\top} X$ is singular).

### Lasso Regression

There is another type of regularized regression called Lasso, where our loss function is the `OLS` loss function plus the absolute value of each coefficient multiplied by some constant - `alpha`:

$$J = \sum_{i=1}^n(y_i - \hat{y_i})^2 + \alpha \sum_{i=1}^n|a_i|$$

`Scikit-learn` also comes with a ready-to-use class for Lasso regression - check it out [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#lasso).

### Feature Importance

Feature importance is the amount of added value that a feature provides to a model when that model is trying to predict the target variable. The more important a feature is, the better it is to be part of a model.

Assessing the feature importance of all features can be used to perform **feature selection** - choosing which features will be part of the final model.

### Lasso Regression and Feature Importance

Lasso regression can actually be used to assess feature importance. This is because **it shrinks the coefficients of less important features to `0`**. The features whose coefficients are not shrunk to `0` are, essentially, selected by the lasso algorithm - when summing them up, **the coefficients act as weights**.

Here's how this can be done in practice:

```python
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
plt.bar(columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()
```

![w03_lasso_coefs_class.png](./assets/w03_lasso_coefs_class.png "w03_lasso_coefs_class.png")

We can see that the most important predictor for our target variable, `blood glucose levels`, is the binary value for whether an individual has `diabetes` or not! This is not surprising, but is a great sanity check.

Benefits:

- allows us to communicate results to non-technical audiences (stakeholders, clients, management);
- helps us eliminate non-important features when we have too many;
- identifies which factors are important predictors for various physical phenomena.

For Lasso there is no simple matrix inverse like Ridge. We'll have to use gradient descent again.

The function we're minimizing is:

$$
\hat{\beta}_{lasso} = \arg \min_{\beta} \Bigg( \| y - X\beta \|^2 + \lambda \sum_{j=1}^{p} |\beta_j| \Bigg), \quad \text{with } \beta_0 \text{ (bias) not regularized.}
$$

So our update step would be:

$$
\beta_j^{(t+1)} = \beta_j^{(t)} - \alpha \Bigg( \frac{1}{m} \sum_{i=1}^m (h_\beta(x_i) - y_i) x_{ij} + \lambda \cdot \text{sign}(\beta_j^{(t)}) \Bigg)
$$

Since the modulo operator is not differentiable at $0$, we'll have to use the sign of the coefficient:

$$
\text{sign}(\beta) =
\begin{cases}
+1 & \text{if } \beta > 0 \\
-1 & \text{if } \beta < 0 \\
\text{any value in } [-1, 1] & \text{if } \beta = 0 & \text{(we'll choose 0 to create the sparcity)}
\end{cases}
$$

## Classification Metrics

### A problem with using `accuracy` always

**Situation:**

A bank contacts our company and asks for a model that can predict whether a bank transaction is fraudulent or not.

Keep in mind that in practice, 99% of transactions are legitimate and only 1% are fraudulent.

> **Definition:** The situation where classes are not equally represented in the data is called ***class imbalance***.

**Problem:**

<details>

<summary>Do you see any problems with using accuracy as the primary metric here?</summary>

The accuracy of a model that predicts every transaction as legitimate is `99%`.

</details>

**Solution:**

<details>

<summary>How do we solve this?</summary>

We have to use other metrics that put focus on the **per-class** performance.

</details>

<details>

<summary>What can we measure then?</summary>

We have to count how the model treats every observation and define the performance of the model based on the number of times that an observation:

- is positive and the model predicts it to be negative;
- or is negative and the model predicts it to be positive;
- or the model predicts its class correctly.

We can store those counts in a table:

![w03_conf_matrix.png](./assets/w03_conf_matrix.png "w03_conf_matrix.png")

> **Definition:** A **confusion matrix** is a table that is used to define the performance of a classification algorithm.

- Across the top are the predicted labels, and down the side are the actual labels.
- Usually, the class of interest is called the **positive class**. As we aim to detect fraud, **the positive class is an *illegitimate* transaction**.
  - The **true positives** are the number of fraudulent transactions correctly labeled;
  - The **true negatives** are the number of legitimate transactions correctly labeled;
  - The **false negatives** are the number of legitimate transactions incorrectly labeled;
  - And the **false positives** are the number of transactions incorrectly labeled as fraudulent.

</details>

**Benefit:**

<details>

<summary>We can retrieve the accuracy. How?</summary>

It's the sum of true predictions divided by the total sum of the matrix.

![w03_cm_acc.png](./assets/w03_cm_acc.png "w03_cm_acc.png")

</details>

<details>

<summary>Do you know what precision is?</summary>

`precision` is the number of true positives divided by the sum of all positive predictions.

- also called the `positive predictive value`;
- in our case, this is the number of correctly labeled fraudulent transactions divided by the total number of transactions classified as fraudulent:

![w03_cm_precision.png](./assets/w03_cm_precision.png "w03_cm_precision.png")

- **high precision** means having a **lower false positive rate**. For our classifier, it means predicting most fraudulent transactions correctly.

$$FPR = \frac{FP}{FP + TN}$$

</details>

<details>

<summary>Do you know what recall is?</summary>

`recall` is the number of true positives divided by the sum of true positives and false negatives

- also called `sensitivity`;

![w03_cm_recall.png](./assets/w03_cm_recall.png "w03_cm_recall.png")

- **high recall** reflects a **lower false negative rate**. For our classifier, this translates to fewer legitimate transactions being classified as fraudulent.

$$FNR = \frac{FN}{TP + FN}$$

Here is a helpful table that can serve as another example:

![w03_example_metrics_2.png](./assets/w03_example_metrics_2.png "w03_example_metrics_2.png")

</details>

<details>

<summary>Do you know what the f1-score is?</summary>

The `F1-score` is the harmonic mean of precision and recall.

- gives equal weight to precision and recall -> it factors in both the number of errors made by the model and the type of errors;
- favors models with similar precision and recall;
- useful when we are seeking a model which performs reasonably well across both metrics.

![w03_cm_f1.png](./assets/w03_cm_f1.png "w03_cm_f1.png")

Another interpretation of the link between precision and recall:

![w03_prec_rec.png](./assets/w03_prec_rec.png "w03_prec_rec.png")

Why is the harmonic mean used? Since both precision and recall are rates (ratios) between `0` and `1`, the harmonic mean helps balance these two metrics by considering their reciprocals. This ensures that a low value in either one has a significant impact on the overall `F1` score, thus incentivizing a balance between the two.

</details>

### Confusion matrix in scikit-learn

We can use the [`confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#confusion-matrix) function in `sklearn.metrics`:

```python
from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)
```

```console
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])
```

We can also use the `from_predictions` static function of the [`ConfusionMatrixDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#confusionmatrixdisplay) class, also in `sklearn.metrics` to plot the matrix:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.tight_layout()
plt.show()
```

![w03_cm_plot.png](./assets/w03_cm_plot.png "w03_cm_plot.png")

We can get the discussed metrics from the confusion matrix, by calling the [`classification_report`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#classification-report) function in `sklearn.metrics`:

```python
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))
```

```console
              precision    recall  f1-score   support

     class 0       0.50      1.00      0.67         1
     class 1       0.00      0.00      0.00         1
     class 2       1.00      0.67      0.80         3

    accuracy                           0.60         5
   macro avg       0.50      0.56      0.49         5
weighted avg       0.70      0.60      0.61         5
```

```python
y_pred = [1, 1, 0]
y_true = [1, 1, 1]
print(classification_report(y_true, y_pred, labels=[1, 2, 3]))
```

```console
              precision    recall  f1-score   support

           1       1.00      0.67      0.80         3
           2       0.00      0.00      0.00         0
           3       0.00      0.00      0.00         0

   micro avg       1.00      0.67      0.80         3
   macro avg       0.33      0.22      0.27         3
weighted avg       1.00      0.67      0.80         3
```

`Support` represents the number of instances for each class within the true labels. If the column with `support` has different numbers, then we have class imbalance.

- `macro average` = $\frac{F1_{class1} + F1_{class2} + F1_{class3}}{3}$
- `weighted average` = $\frac{F1_{class1}*SUPPORT_{class1} + F1_{class2}*SUPPORT_{class2} + F1_{class3}*SUPPORT_{class3}}{3}$
- `micro average` = $\frac{F1_{class1}*SUPPORT_{class1} + F1_{class2}*SUPPORT_{class2} + F1_{class3}*SUPPORT_{class3}}{SUPPORT_{class1} + SUPPORT_{class2} + SUPPORT_{class3}}$

## The receiver operating characteristic curve (`ROC` curve)

<details>

<summary>Do you know what the ROC curve shows?</summary>

> **Note:** Use the `ROC` curve only when doing ***binary* classification**.

The default probability threshold for logistic regression in scikit-learn is `0.5`. What happens as we vary this threshold?

We can use a receiver operating characteristic, or ROC curve, to visualize how different thresholds affect `true positive` and `false positive` rates.

![w04_roc_example.png](./assets/w04_roc_example.png "w04_roc_example.png")

The dotted line represents a random model - one that randomly guesses labels.

When the threshold:

- equals `0` (`p=0`), the model predicts `1` for all observations, meaning it will correctly predict all positive values, and incorrectly predict all negative values;
- equals `1` (`p=1`), the model predicts `0` for all data, which means that both true and false positive rates are `0` (nothing gets predicted as positive).

![w04_roc_edges.png](./assets/w04_roc_edges.png "w04_roc_edges.png")

If we vary the threshold, we get a series of different false positive and true positive rates.

![w04_roc_vary.png](./assets/w04_roc_vary.png "w04_roc_vary.png")

A line plot of the thresholds helps to visualize the trend.

![w04_roc_line.png](./assets/w04_roc_line.png "w04_roc_line.png")

<details>

<summary>What plot would be produced by the perfect model?</summary>

One in which the line goes straight up and then right.

![w04_perfect_roc.png](./assets/w04_perfect_roc.png "w04_perfect_roc.png")

</details>

</details>

### In `scikit-learn`

In scikit-learn the `roc_curve` is implemented in the `sklearn.metrics` module.

```python
import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

fpr
```

```console
array([0. , 0. , 0.5, 0.5, 1. ])
```

```python
tpr
```

```console
array([0. , 0.5, 0.5, 1. , 1. ])
```

```python
thresholds
```

```console
array([ inf, 0.8 , 0.4 , 0.35, 0.1 ])
```

To plot the curve can create a `matplotlib` plot or use a built-in function.

Using `matplotlib` would look like this:

```python
plt.plot([0, 1], [0, 1], 'k--') # to draw the dashed line
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
```

![w04_example_roc_matlotlib.png](./assets/w04_example_roc_matlotlib.png "w04_example_roc_matlotlib.png")

We could also use the `from_predictions` function in the `RocCurveDisplay` class to create plots.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y, scores)
plt.tight_layout()
plt.show()
```

![w04_example_roc_display.png](./assets/w04_example_roc_display.png "w04_example_roc_display.png")

<details>

<summary>The above figures look great, but how do we quantify the model's performance based on them?</summary>

### The Area Under the Curve (`AUC`)

If we have a model with `true_positive_rate=1` and `false_positive_rate=0`, this would be the perfect model.

Therefore, we calculate the area under the ROC curve, a **metric** known as `AUC`. Scores range from `0` to `1`, with `1` being ideal.

In the below figure, the model scores `0.67`, which is `34%` better than a model making random guesses.

![w04_example_roc_improvement.png](./assets/w04_example_roc_improvement.png "w04_example_roc_improvement.png")

### In `scikit-learn`

In scikit-learn the area under the curve can be calculated in two ways.

Either by using the `RocCurveDisplay.from_predictions` function:

![w04_example_roc_display_note.png](./assets/w04_example_roc_display_note.png "w04_example_roc_display_note.png")

or by using the `roc_auc_score` function in the `sklearn.metrics` module:

```python
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))
```

```console
0.6700964152663693
```

</details>

## Cross Validation

Currently, we're using train-test split to compute model performance.

<details>

<summary>What are the potential downsides of using train-test split?</summary>

1. Model performance is dependent on the way we split up the data: we may get different results if we do another split.
2. The data points in the test set may have some peculiarities: the R-squared computed on it is not representative of the model's ability to generalize to unseen data.
3. The points in the test set will never be used for training the model: we're missing out on potential benefits.

</details>

<details>

<summary>Have you heard of the technique called cross-validation?</summary>

It is a vital approach to evaluating a model. It maximizes the amount of data that is available to the model, as the model is not only trained but also tested on all of the available data.

Here's a visual example of what cross-validation comprises of:

![w03_cv_example1.png](./assets/w03_cv_example1.png "w03_cv_example1.png")

We begin by splitting the dataset into `k` groups or folds - ex. `5`. Then we set aside the first fold as a test set, fit our model on the remaining four folds, predict on our test set, and compute the metric of interest, such as R-squared.

Next, we set aside the second fold as our test set, fit on the remaining data, predict on the test set, and compute the metric of interest.

![w03_cv_example2.png](./assets/w03_cv_example2.png "w03_cv_example2.png")

Then similarly with the third fold, the fourth fold, and the fifth fold. As a result we get five values of R-squared from which we can compute statistics of interest, such as the mean, median, and 95% confidence intervals.

![w03_cv_example3.png](./assets/w03_cv_example3.png "w03_cv_example3.png")

Usually the value for `k` is either `5` or `10`.

</details>

<details>

<summary>What is the trade-off of using cross-validation compared to train-test split?</summary>

Using more folds is more computationally expensive. This is because we're fitting and predicting multiple times, instead of just `1`.

</details>

To perform k-fold cross-validation in `scikit-learn`, we can use the function [`cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#cross-val-score) and the class [`KFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#kfold) that are part of `sklearn.model_selection`.

- the `KFold` class allows us to set a seed and shuffle our data, making our results repeatable downstream. The `n_splits` argument has a default of `5`, but in this case we assign `2`, allowing us to use `2` folds from our dataset for cross-validation. We also set `shuffle=True`, which shuffles our dataset **before** splitting into folds. We assign a seed to the `random_state` keyword argument, ensuring our data would be split in the same way if we repeat the process making the results repeatable downstream. We save this as the variable `kf`.

```python
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])

kf = KFold(n_splits=2, shuffle=True, random_state=42)

print(list(kf.split(X)))
print(list(kf.split(y)))
```

```console
[(array([0, 2]), array([1, 3])), (array([1, 3]), array([0, 2]))]
[(array([0, 2]), array([1, 3])), (array([1, 3]), array([0, 2]))]
```

The result is a list of tuples of arrays with training and testing indices. In this case, we would use elements at indices `0` and `2` to train a model and evaluate it on elements at indices `1` and `3`.

- in practice, you wouldn't call `kf.split` directly. Instead, you would pass the `kf` object to `cross_val_score`. It accepts a model, feature data and target data as the first three positional arguments. We also specify the number of folds by setting the keyword argument `cv` equal to our `kf` variable.

```python
cv_results = cross_val_score(linear_reg, X, y, cv=kf)
print(cv_results)

# we can calculate the 95% confidence interval passing our results followed by a list containing the upper and lower limits of our interval as decimals 
print(np.quantile(cv_results, [0.025, 0.975]))
```

```console
[0.70262578, 0.7659624, 0.75188205, 0.76914482, 0.72551151, 0.736]
array([0.7054865, 0.76874702])
```

This returns an array of cross-validation scores, which we assign to `cv_results`. The length of the array is the number of folds utilized.

> **Note:** the reported score is the result of calling `linear_reg.score`. Thus, when the model is linear regression, the score reported is $R^2$.

## Hyperparameter optimization/tuning

### Hyperparameters

<details>

<summary>What are hyperparameters?</summary>

A hyperparameter is a variable used for selecting a model's parameters.

</details>

<details>

<summary>What are some examples?</summary>

- $a$ in `Ridge`;
- $k$ in `KNN`.

</details>

### Introduction

Recall that we had to choose a value for `alpha` in ridge and lasso regression before fitting it.

Likewise, before fitting and predicting `KNN`, we choose `n_neighbors`.

> **Definition:** Parameters that we specify before fitting a model, like `alpha` and `n_neighbors`, are called **hyperparameters**.

<details>

<summary>So, a fundamental step for building a successful model is choosing the correct hyperparameters. What's the best way to go about achieving this?</summary>

We can try lots of different values, fit all of them separately, see how well they perform, and choose the best values!

> **Definition:** The process of trying out different hyperparameters until a satisfactory performance threshold is reached is called **hyperparameter tuning**.

</details>

When fitting different hyperparameter values, we use cross-validation to avoid overfitting the hyperparameters to the test set:

- we split the data into train and test;
- and perform cross-validation on the training set.

We withhold the test set and use it only for evaluating the final, tuned model.

> **Definition:** The set on which the model is evaluated on during cross-validation is called the **validation set**.

Notice how:

- the training set is used to train the model;
- the validation set is used to tune the model until a satisfactory performance threshold is used;
- the test set is used as the final set on which model performance is reported. It is data that the model hasn't seen, but because it is labeled, we can use to evaluate the performance eliminating bias.

### Grid search cross-validation

<details>

<summary>Do you know what grid search cross-validation comprises of?</summary>

One approach for hyperparameter tuning is called **grid search**, where we choose a grid of possible hyperparameter values to try. In Python this translates to having a dictionary that maps strings to lists/arrays of possible values to choose from.

For example, we can search across two hyperparameters for a `KNN` model - the type of metric and a different number of neighbors. We perform `k`-fold cross-validation for each combination of hyperparameters. The mean scores (in this case, accuracies) for each combination are shown here:

![w04_grid_search_cv.png](./assets/w04_grid_search_cv.png "w04_grid_search_cv.png")

We then choose hyperparameters that performed best:

![w04_grid_search_cv_choose_best.png](./assets/w04_grid_search_cv_choose_best.png "w04_grid_search_cv_choose_best.png")

To get a list of supported values for the model we're building, we can use the scikit-learn documentation. For example, in the documentation for [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#logisticregression), the possible values for the `solver` parameter can be seen as one scrolls down the page.

### In `scikit-learn`

`scikit-learn` has an implementation of grid search using cross-validation in the `sklearn.model_selection` module:

```python
from sklearn.model_selection import GridSearchCV

# instantiate a `KFold` object

param_grid = {
    'alpha': np.arange(0.0001, 1, 10),
    'solver': ['sag', 'lsqr']
}

ridge_cv = GridSearchCV(Ridge(), param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```

```console
{'alpha': 0.0001, 'solver': 'sag'}
0.7529912278705785
```

</details>

<details>

<summary>What is the main problem of grid search?</summary>

Grid search is great - it allows us to scan a predefined parameter space fully. However, it does not scale well:

<details>

<summary>How many fits will be done while performing 3-fold cross-validation for 1 hyperparameter with 10 values?</summary>

Answer: 30.

</details>

<details>

<summary>How many fits will be done while performing 10-fold cross-validation for 3 hyperparameters with 10 values each?</summary>

Answer: 10,000!

We can verify this:

```python
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV


def main():
    iris = datasets.load_iris()

    parameters = {
        'degree': np.arange(10),
        'C': np.linspace(0, 10, 10),
        'tol': np.linspace(0.0001, 0.01, 10),
    }

    print(len(parameters['degree']))
    print(len(parameters['C']))
    print(len(parameters['tol']))

    svc = svm.SVC() # This is a support vector machine. We'll talk about it soon.
    clf = GridSearchCV(svc, parameters, cv=10, verbose=1)
    clf.fit(iris.data, iris.target)
    print(sorted(clf.cv_results_))


if __name__ == '__main__':
    main()
```

This is because:

1. The total number of parameter combinations is `10^3 = 1000` (we have one for-loop with two nested ones inside).

  ```python
  # pseudocode
  for d in degrees:
    for c in cs:
      for tol in tols:
        # this is one combination
  ```

2. For every single one combination we do a `10`-fold cross-validation to get the mean metric. This means that every single one of the paramter combinations is the same while we shift the training and testing sets `10` times.

</details>

<details>

<summary>What is the formula in general then?</summary>

```text
number of fits = number of folds * number of total hyperparameter values
```

</details>

<details>

<summary>How can we go about solving this?</summary>

### Randomized search cross-validation

We can perform a random search, which picks random hyperparameter values rather than exhaustively searching through all options.

```python
from sklearn.model_selection import RandomizedSearchCV

# instantiate a `KFold` object

param_grid = {
    'alpha': np.arange(0.0001, 1, 10),
    'solver': ['sag', 'lsqr']
}

# optionally set the "n_iter" argument, which determines the number of hyperparameter values tested (default is 10)
# 5-fold cross-validation with "n_iter" set to 2 performs 10 fits
ridge_cv = RandomizedSearchCV(Ridge(), param_grid, cv=kf, n_iter=2)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```

In this case it is able to find the best hyperparameters from our previous grid search!

```console
{'alpha': 0.0001, 'solver': 'sag'}
0.7529912278705785
```

#### Benefits

This allows us to search from large parameter spaces efficiently.

### Evaluating on the test set

We can evaluate model performance on the test set by passing it to a call of the grid/random search object's `.score` method.

```python
test_score = ridge_cv.score(X_test, y_test)
test_score
```

```console
0.7564731534089224
```

</details>

</details>

# Week 05 - Preprocessing and Pipelines. Support Vector Machines

## Preprocessing and Pipelines

### Dealing with categorical features

`scikit-learn` requires data that:

- is in **numeric** format;
- has **no missing** values.

All the data that we have used so far has been in this format. However, with real-world data:

- this will rarely be the case;
- typically we'll spend around 80% of our time solely focusing on preprocessing it before we can build models (may come as a shoker).

Say we have a dataset containing categorical features, such as `color` and `genre`. Those features are not numeric and `scikit-learn` will not accept them.

<details>

<summary>How can we solve this problem?</summary>

We can substitute the strings with numbers.

<details>

<summary>What approach can we use to do this?</summary>

We need to convert them into numeric features. We can achieve this by **splitting the features into multiple *binary* features**:

- `0`: observation was not that category;
- `1`: observation was that category.

![w05_dummies.png](./assets/w05_dummies.png "w05_dummies.png")

> **Definition:** Such binary features are called **dummy variables**.

We create dummy features for each possible `genre`. As each song has one `genre`, each row will have a `1` in only one of the ten columns and `0` in the rest.

**Benefit:** We can now pass categorical features to models as well.

</details>

</details>

<details>

<summary>What is one problem of this approach?</summary>

#### Dropping one of the categories per feature

If a song is not any of the first `9` genres, then implicitly, it is a `Rock` song. That means we only need nine features, so we can delete the `Rock` column.

If we do not do this, we are duplicating information, which might be an issue for some models (we're essentially introducing linear dependence - if I know the values for the first `9` columns, I for sure know the value of the `10`-th one as well).

![w05_dummies.png](./assets/w05_dummies_drop_first.png "w05_dummies.png")

Let's see why we have linear dependence:

To check for linear dependece, we can see what happens when we do one-hot encoding:

$$x_2 = 1 - x_0 - x_1 = (1, 1, 1) - (0, 1, 1) = (1, 0, 0)$$

This means that $x_2$ is linearly dependent hence brings no new information.

</details>

#### In `scikit-learn` and `pandas`

To create dummy variables we can use:

- the `OneHotEncoder` class if we're working with `scikit-learn`;
- or `pandas`'s `get_dummies` function.

We will use `get_dummies`, passing the categorical column.

```python
df_music.head()
```

```console
   popularity  acousticness  danceability  duration_ms  energy  instrumentalness  liveness  loudness  speechiness       tempo  valence       genre
0          41        0.6440         0.823       236533   0.814          0.687000    0.1170    -5.611       0.1770  102.619000    0.649        Jazz
1          62        0.0855         0.686       154373   0.670          0.000000    0.1200    -7.626       0.2250  173.915000    0.636         Rap
2          42        0.2390         0.669       217778   0.736          0.000169    0.5980    -3.223       0.0602  145.061000    0.494  Electronic
3          64        0.0125         0.522       245960   0.923          0.017000    0.0854    -4.560       0.0539  120.406497    0.595        Rock
4          60        0.1210         0.780       229400   0.467          0.000134    0.3140    -6.645       0.2530   96.056000    0.312         Rap
```

```python
# As we only need to keep nine out of our ten binary features, we can set the "drop_first" argument to "True".
music_dummies = pd.get_dummies(df_music['genre'], drop_first=True)
music_dummies.head()
```

```console
   Anime  Blues  Classical  Country  Electronic  Hip-Hop   Jazz    Rap   Rock
0  False  False      False    False       False    False   True  False  False
1  False  False      False    False       False    False  False   True  False
2  False  False      False    False        True    False  False  False  False
3  False  False      False    False       False    False  False  False   True
4  False  False      False    False       False    False  False   True  False
```

Printing the first five rows, we see pandas creates `9` new binary features. The first song is `Jazz`, and the second is `Rap`, indicated by a `True`/`1` in the respective columns.

```python
music_dummies = pd.get_dummies(df_music['genre'], drop_first=True, dtype=int)
music_dummies.head()
```

```console
   Anime  Blues  Classical  Country  Electronic  Hip-Hop  Jazz  Rap  Rock
0      0      0          0        0           0        0     1    0     0
1      0      0          0        0           0        0     0    1     0
2      0      0          0        0           1        0     0    0     0
3      0      0          0        0           0        0     0    0     1
4      0      0          0        0           0        0     0    1     0
```

To bring these binary features back into our original DataFrame we can use `pd.concat`, passing a list containing the music DataFrame and our dummies DataFrame, and setting `axis=1`. Lastly, we can remove the original genre column using `df.drop`, passing the `columns=['genre']`.

```python
music_dummies = pd.concat([df_music, music_dummies], axis=1)
music_dummies = music_dummies.drop(columns=['genre'])
```

If the DataFrame only has one categorical feature, we can pass the entire DataFrame, thus skipping the step of combining variables.

If we don't specify a column, the new DataFrame's binary columns will have the original feature name prefixed, so they will start with `genre_`.

```python
music_dummies = pd.get_dummies(df_music, drop_first=True)
music_dummies.columns
```

```console
Index(['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo',   
       'valence', 'genre_Anime', 'genre_Blues', 'genre_Classical',
       'genre_Country', 'genre_Electronic', 'genre_Hip-Hop', 'genre_Jazz',   
       'genre_Rap', 'genre_Rock'],
      dtype='object')
```

Notice the original genre column is automatically dropped. Once we have dummy variables, we can fit models as before.

### EDA with categorical feature

We will be working with the above music dataset this week, for both classification and regression problems.

Initially, we will build a regression model using all features in the dataset to predict song `popularity`. There is one categorical feature, `genre`, with ten possible values.

We can use a `boxplot` to visualize the relationship between categorical and numeric features:

![w05_eda.png](./assets/w05_eda.png "w05_eda.png")

### Handling missing data

<details>

<summary>How can we define missing data?</summary>

When there is no value for a feature in a particular row, we call it missing data.

</details>

<details>

<summary>Why might this happen?</summary>

- there was no observation;
- the data might be corrupt;
- the value is invalid;
- etc, etc.

</details>

<details>

<summary>What pandas functions/methods can we use to check how much of our data is missing?</summary>

We can use the `isna()` pandas method:

```python
# get the number of missing values per column
df_music.isna().sum().sort_values(ascending=False)
```

```console
acousticness        200
energy              200
valence             143
danceability        143
instrumentalness     91
duration_ms          91
speechiness          59
tempo                46
liveness             46
loudness             44
popularity           31
genre                 8
dtype: int64
```

We see that each feature is missing between `8` and `200` values!

Sometimes it's more appropriate to see the percentage of missing values:

```python
# get the number of missing values per column
df_music.isna().mean().sort_values(ascending=False)
```

```console
acousticness        0.200
energy              0.200
valence             0.143
danceability        0.143
instrumentalness    0.091
duration_ms         0.091
speechiness         0.059
tempo               0.046
liveness            0.046
loudness            0.044
popularity          0.031
genre               0.008
dtype: float64
```

</details>

<details>

<summary>How could we handle missing data in your opinion?</summary>

1. Remove it.
2. Substitute it with a plausible value.

</details>

<details>

<summary>What similar analysis could we do to find columns that are not useful?</summary>

We can check the number of unique values in categorical columns. If every row has a unique value, then this feature is useless - there is no pattern.

</details>

#### Removing missing values

Two common approaches:

- for the columns with $< 5\%$ missing values, remove the **rows** that contain those missing values.
- for the columns with $> 65\%$ missing values, remove the **columns**.

To remove observations with missing values from a certain columnset, we can use the pandas `dropna` method, passing a list of columns to the `subset` argument. The idea being, that if there are missing values in our subset column, **the entire row** is removed.

```python
df_music = df_music.dropna(subset=['genre', 'popularity', 'loudness', 'liveness', 'tempo'])
df_music.isna().mean().sort_values(ascending=False)
```

```console
acousticness        0.199552
energy              0.199552
valence             0.142377
danceability        0.142377
speechiness         0.059417
duration_ms         0.032511
instrumentalness    0.032511
popularity          0.000000
loudness            0.000000
liveness            0.000000
tempo               0.000000
genre               0.000000
dtype: float64
```

Other rules of thumb include:

- removing every missing value from the target feature;
- removing columns whose missing values are above `65%`;
- etc, etc.

#### Imputing missing values

> **Definition:** Making an educated guess as to what the missing values could be.

Which value to use?

- for numeric features, it's best to use the `median` of the column;
- for categorical values, we typically use the `mode` - the most frequent value.

<details>

<summary>What should we do to our data before imputing missing values?</summary>

We must split our data before imputing to avoid leaking test set information to our model, a concept known as **data leakage**.

</details>

Here is a workflow for imputation:

```python
from sklearn.impute import SimpleImputer
imp_cat = SimpleImputer(strategy='most_frequent')
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)
```

For our numeric data, we instantiate and use another imputer.

```python
imp_num = SimpleImputer(strategy='median') # note that default is 'mean'
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)
X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)
```

> **Definition:** Due to their ability to transform our data, imputers are known as **transformers**.

#### Using pipelines

> **Definition:** A pipeline is an object used to run a series of transformers and build a model in a single workflow.

```python
from sklearn.pipeline import Pipeline

df_music = df_music.dropna(subset=['genre', 'popularity', 'loudness', 'liveness', 'tempo'])
df_music['genre'] = np.where(df_music['genre'] == 'Rock', 1, 0)
X = df_music.drop(columns=['genre'])
y = df_music['genre']
```

To build a pipeline we construct a list of steps containing tuples with the step names specified as strings, and instantiate the transformer or model.

> **Note:** In a pipeline, each step but the last must be a transformer.

```python
steps = [('imputation', SimpleImputer()),
('logistic_regression', LogisticRegression())]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```

### Centering and scaling

Let's use the `.describe().T` function composition to check out the ranges of some of our feature variables in the music dataset.

![w05_scaling_problem.png](./assets/w05_scaling_problem.png "w05_scaling_problem.png")

We see that the ranges vary widely:

- `duration_ms` ranges from `-1` to `1.6` million;
- `speechiness` contains only decimal values;
- `loudness` only has negative values.

<details>

<summary>What is the problem here?</summary>

Some machine learning models use some form of distance to inform them, so if we have features on far larger scales, they can disproportionately influence our model.

For example, `KNN` uses distance explicitly when making predictions.

</details>

<details>

<summary>What are the possible solutions?</summary>

We actually want features to be on a similar scale. To achieve this, we can `normalize` or `standardize` our data, often also referred to as scaling and centering.

As benefits we get:

1. Model agnostic data, meaning that any model would be able to work with it.
2. All features have equal meaning/contribution/weight.

</details>

### Standardization and normalization

Given any column, we can subtract the mean and divide by the variance:

![w05_standardization_formula.png](./assets/w05_standardization_formula.png "w05_standardization_formula.png")

- Result: All features are centered around `0` and have a variance of `1`.
- Terminology: This is called **standardization**.

We can also subtract the minimum and divide by the range of the data:

![w05_normalization_formula.png](./assets/w05_normalization_formula.png "w05_normalization_formula.png")

- Result: The normalized dataset has minimum of `0` and maximum of `1`.
- This is called **normalization**.

Or, we can center our data so that it ranges from `-1` to `1` instead. In general to get a value in a new interval `[a, b]` we can use the formula:

$$x''' = (b-a)\frac{x - \min{x}}{\max{x} - \min{x}} + a$$

### Scaling in `scikit-learn`

To scale our features, we can use the `StandardScaler` class from `sklearn.preprocessing`:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(np.mean(X), np.std(X))
print(np.mean(X_train_scaled), np.std(X_train_scaled))
```

```console
19801.42536120538, 71343.52910125865
2.260817795600319e-17, 1.0
```

Looking at the mean and standard deviation of the columns of both the original and scaled data verifies the change has taken place.

We can also put a scaler in a pipeline!

```python
steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps).fit(X_train, y_train)
```

and we can use that pipeline in cross validation. When we specify the hyperparameter space the dictionary has keys that are formed by the pipeline step name followed by a double underscore, followed by the hyperparameter name. The corresponding value is a list or an array of the values to try for that particular hyperparameter.

In this case, we are tuning `n_neighbors` in the `KNN` model:

```python
pipeline = Pipeline(steps)
parameters = {'knn__n_neighbors': np.arange(1, 50)}
```

### How do we decide which model to try out in the first place?

1. The size of our dataset.

   - Fewer features = a simpler model and can reduce training time.
   - Some models, such as Artificial Neural Networks, require a lot of data to perform well.

2. Interpretability.

   - Some models are easier to explain which can be important for stakeholders.
   - Linear regression has high interpretability as we can understand the coefficients.

3. Flexibility.

   - More flexibility = higher accuracy, because fewer assumptions are made about the data.
   - A KNN model does not assume a linear relationship between the features and the target.

4. Train several models and evaluate performance out of the box (i.e. without hyperparameter tuning)

   - Regression model performance
     - RMSE
     - $\text{adjusted} R^2$

   - Classification model performance
     - Confusion matrix
     - Precision, Recall, F1-score
     - ROC AUC

5. Scale the data

    Recall that the performance of some models is affected by scaling our data. Therefore, it is generally best to scale our data before evaluating models out of the box.

    Models affected by scaling:

    - KNN;
    - Linear Regression (+ Ridge, Lasso);
    - Logistic Regression;
    - etc, etc, in general, every model that uses distance when predicting or has internal logic that works with intervals (activation functions in NN)

    Models not affected by scaling:

    - Decision Trees;
    - Random Forest;
    - XGBoost;
    - Catboost;
    - etc, etc, in general, models that are based on trees.

## Support Vector Machines

### The use-case for kernel SVMs

Let's say that during our data audit we find two features that can be used to perfectly separate our training samples:

![w05_ksvm0.png](assets/w05_ksvm0.png "w05_ksvm0.png")

<details>

<summary>If we had the ability to choose between a straight-line model and a curvy-line model, which one would we choose?</summary>

We utilize [Occam's razor](https://en.wikipedia.org/wiki/Occam%27s_razor) and **choose the simpler case**: straight-line model.

</details>

<details>

<summary>What are the possible models, then, that can solve this problem?</summary>

We have many possibilities - here're some:

![w05_ksvm1.png](assets/w05_ksvm1.png "w05_ksvm1.png")

</details>

But, clearly, out of all possible such models, we know that is one and only one that is the best for the given data, right?

<details>

<summary>Which one is it?</summary>

It'd be the one that **maximizes the width** of the "street" between the positive and negative samples:

![w05_ksvm2.png](assets/w05_ksvm2.png "w05_ksvm2.png")

</details>

This is the problem that SVMs solve - they find the model that best separates the training examples!

### Prediction function

Let's say that we do decide to go with an SVM for the above problem. What would be our logic for classifying this new sample:

![w05_ksvm3.png](assets/w05_ksvm3.png "w05_ksvm3.png")

<details>

<summary>Reveal answer</summary>

It'd be really nice if we could base our decision on the median of the street and say that the class would be `circle` if `x` is on the right side of the median / street.

To check that, it'd be easiest to check whether the **dot product** between `x` and an orthogonal vector to the street, ex. `w`, is greater than a constant `c`:

![w05_ksvm4.png](assets/w05_ksvm4.png "w05_ksvm4.png")

$$w^T x >= b$$

And if we let $c = -b$, then we get our familiar linear combination:

$$w^T x + b >= 0$$

</details>

And this is our model. We can notice that similar to `KNN`, the `SVM` **does *not* output probabilities**. Instead, it outputs either `0` or `1` for the two classes. Unlike `KNN`, but similar to `LogisticRegression`, the `SVM` is typically used for **binary classification**.

Our goal now is to find the parameters of the model - $w$ and $b$:

- we don't know what constant to use, because it depends on $w$;
- but there are many orthogonal vectors $w$ because it can be of any length: how do we choose one?

<details>

<summary>Do you know what strategy support vector machines employ?</summary>

The strategy used by SVMs is that of **enforcing enough *constraints* on the decision function** so that $w$ and $b$ can be uniquely determined. More specifically, two constraints are placed.

</details>

<details>

<summary>What might they be?</summary>

1. For every positive training sample, we want our decision function to output at least $1$:

$$w^T x_+ + b >= 1$$

2. For every negative training sample, we want our decision function to output at most $-1$:

$$w^T x_- + b <= -1$$

</details>

Carrying around two equations like this will be a pain.

<details>

<summary>Can we use the approach we showed last time to combine them into one?</summary>

Yep - we can use the two class labels! If we map them to $-1$ (for all negative samples) and $+1$ (for all positive samples) and multiply the left side of each equation, we'll get that:

Constraint 1 stays the same since $y_i$ is $+1$:

$$y_i (w^T x_i + b) >= 1$$

Constraint 2 becomes:

$$-y_i (w^T x_i + b) <= -1$$

Multiplying by $-1$ on both sides to remove the negative signs:

$$y_i (w^T x_i + b) >= 1$$

Oh - two equations are the same! So, in the end our constraint ends up being:

$$y_i (w^T x_i + b) - 1 >= 0$$

</details>

<details>

<summary>What sign would the value, outputted by the linear combination, need to have in order for the model to satisfy the constraint for negative samples?</summary>

Since $y_i$ is $-1$, it'd need to be negative so that the expression on the left becomes positive.

</details>

<details>

<summary>What sign would the value, outputted by the linear combination, need to have in order for the model to satisfy the constraint for the samples on the edges of street?</summary>

For samples on the edge of the street, we'd have to output $0$.

</details>

<details>

<summary>Do you know how these samples are called?</summary>

Support vectors! Essentially, these are the only samples we need to pay attention to as they define the decision boundary.

</details>

### Fitting the model

Given the constraint $y_i (w^T x_i + b) - 1 >= 0$, we can in fact find the optimal $w$ and $b$. Let's start thinking about $w$.

<details>

<summary>What was its geometrical interpretation?</summary>

It is the vector that allows us to calculate the width of the "street" - the distance between the positive and negative samples.

</details>

Great! So, if we take the following to samples:

![w05_ksvm5.png](assets/w05_ksvm5.png "w05_ksvm5.png")

<details>

<summary>What would be the formula for the width of the street?</summary>

![w05_ksvm6.png](assets/w05_ksvm6.png "w05_ksvm6.png")

$$\text{width} = (x_+ - x_-) \cdot \frac{w}{\|w\|}$$

</details>

<details>

<summary>We aren't done yet - how can we expand this expression using the meaning of support vectors?</summary>

Firstly, we can expand the parenthesis:

$$\text{width} = (x_+ - x_-) \cdot \frac{w}{\|w\|} = \frac{x_+ w - x_- w}{\|w\|}$$

And then we can use the property that $y_i (w^T x_i + b) - 1 = 0$ for all support vectors:

- For $x_+$:

   $$y_+ (w^T x_+ + b) - 1 = 0$$

   Since $y_+ = +1$:

   $$w^T x_+ + b - 1 = 0$$

   Moving $+ b - 1$ to the other side:

   $$w^T x_+ = 1 - b$$

- For $x_-$:

   $$y_- (w^T x_- + b) - 1 = 0$$

   Since $y_- = -1$:

   $$- w^T x_- - b - 1 = 0$$

   Moving $- b - 1$ to the other side:

   $$- w^T x_- = 1 + b$$

Substituting with these equalities in the above equation, we get that the width is inversely proportional to the magnitude of $w$:

$$\text{width} = \frac{x_+ w - x_- w}{\|w\|} = \frac{1 - b + 1 + b}{\|w\|} = \frac{2}{\|w\|}$$

</details>

<details>

<summary>What is our goal then?</summary>

$$\text{max} \frac{2}{\|w\|}$$

</details>

<details>

<summary>What is that equivalent to?</summary>

$$\text{max} \frac{1}{\|w\|}$$

</details>

<details>

<summary>What is that equivalent to?</summary>

$$\text{min} \|w\|$$

</details>

<details>

<summary>What is that equivalent to?</summary>

$$\text{min} \frac{1}{2} \|w\|^2$$

</details>

And it this point, we have our loss function:

$$\text{min } J(\theta) = \frac{1}{2} \|w\|^2$$

$$\text{s.t. } y_i (w^T x_i + b) - 1 >= 0$$

<details>

<summary>Could we solve this with gradient descent?</summary>

This is a **constrained convex optimization problem**. Gradient descent doesn't naturally handle constraints.

Instead, we use [**Lagrange multipliers**](https://en.wikipedia.org/wiki/Lagrange_multiplier) to incorporate the constraints into the objective function, leading us to **the dual SVM problem**.

</details>

The introduction of Lagrange multipliers is expressed as the introduction of new coefficients $\alpha_i$, $\alpha_i \isin [0, \inf]$ ($i$ = for each of the samples) that will be used to enforce the constraints of the primal problem:

$$\max_{\alpha_i >= 0} L(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_i \alpha_i \left[ y_i (w^T x_i + b) - 1 \right]$$

With this addition, our optimization problem becomes a maximization (adversary) minimization (us) one:

- Suppose the constraint is violated, ex. $y_i (w^T x_i + b) - 1 = -1$ ($-1 < 0$) - what value would the adversary place on $\alpha$?

   <details>

   <summary>Reveal answer</summary>

   A really large positive value, since they'd have $-\alpha (-1)$, so: $\alpha = \inf$.

   </details>

- Suppose the constraint is not violated, ex. $100$ ($100 > 0$) - what value would the adversary place on $\alpha$?

   <details>

   <summary>Reveal answer</summary>

   The lowest they can - $0$.

   </details>

Ok - great - we now have to find the extremum of a function!

<details>

<summary>What was the process for that?</summary>

We take the derivatives and set them to $0$!

- The derivative of $L$ with respect to $w$ is:

   $$\frac{\partial L}{\partial w} = w - \sum_i \alpha_i y_i x_i$$

   We set it to $0$:

   $$w - \sum_i \alpha_i y_i x_i = 0$$

   And get that $w$ is just a linear sum of the samples:

   $$w = \sum_i \alpha_i y_i x_i$$

- The derivative of $L$ with respect to $b$ is:

   $$\frac{\partial L}{\partial b} = - \sum_i \alpha_i y_i$$

   We set it to $0$ and get that the dot product between the $\alpha$ vector and the $b$ vector is $0$:

   $$\sum_i \alpha_i y_i = 0$$

> **Note:** This $\sum_i \alpha_i y_i = 0$ is actually a constraint on the possible values for $\alpha$. It is referred to as the "linear equality constraint". We'll have to enforce this to our optimization problem.

</details>

<details>

<summary>What would our next step be?</summary>

Let's substitute $w$ with what we've found.

We can rewrite the norm of $w$ as a matrix operation:

$$\max_{\alpha_i >= 0} L = \frac{1}{2} w^T w - \sum_i \alpha_i \left[ y_i (w^T x_i + b) - 1 \right]$$

Let's now distribute the $y$s in the parenthesis:

$$\max_{\alpha_i >= 0} L = \frac{1}{2} w^T w - \sum_i \alpha_i \left( y_i w^T x_i + y_i b - 1 \right)$$

Let's now distribute the $\alpha$s in the parenthesis:

$$\max_{\alpha_i >= 0} L = \frac{1}{2} w^T w - \sum_i \left( \alpha_i y_i w^T x_i + \alpha_i y_i b - \alpha_i \right)$$

Distributing the sum:

$$\max_{\alpha_i >= 0} L = \frac{1}{2} w^T w - \sum_i \alpha_i y_i w^T x_i - \sum_i \alpha_i y_i b + \sum_i \alpha_i$$

Moving constants outside of the sum:

$$\max_{\alpha_i >= 0} L = \frac{1}{2} w^T w - w^T \sum_i \alpha_i y_i x_i - b \sum_i \alpha_i y_i + \sum_i \alpha_i$$

We can remove one element since $\sum_i \alpha_i y_i = 0$ and start substituting $w$:

$$\max_{\alpha_i >= 0} L = \frac{1}{2} w^T \sum_i \alpha_i y_i x_i - w^T \sum_i \alpha_i y_i x_i + \sum_i \alpha_i$$

We have the same multipliers for $w^T$, so we're left with $-\frac{1}{2}$:

$$\max_{\alpha_i >= 0} L = - \frac{1}{2} w^T \sum_i \alpha_i y_i x_i + \sum_i \alpha_i$$

</details>

Let's substitute the second $w$ and change the indices of the sums so as to make it clear that they are separate:

$$\max_{\alpha_i >= 0} L = - \frac{1}{2} (\sum_i \alpha_i y_i x_i^T) (\sum_j \alpha_j y_j x_j) + \sum_i \alpha_i$$

Combining the sums yields:

$$\max_{\alpha_i >= 0} L(\alpha) = - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^T x_j + \sum_i \alpha_i$$

Which we can make into a minimization problem:

$$\min_{\alpha_i >= 0} L(\alpha) = \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^T x_j - \sum_i \alpha_i$$

<details>

<summary>Are we done?</summary>

Not yet - we need to enforce the constraints.

</details>

<details>

<summary>What variables would the constraints be related to?</summary>

Since this is a function the $\alpha$s, they are the only variable that can vary, so it'd be the $\alpha$s.

</details>

<details>

<summary>Which is the first constraint?</summary>

The linear equality constraint that we got from differentiation: $\sum_i \alpha_i y_i = 0$.

</details>

The second one is related to the possible values values that the $\alpha$s can take.

<details>

<summary>What values did we say that they can take initially?</summary>

$$[0, \inf]$$

</details>

<details>

<summary>What do very high values mean?</summary>

The loss will be higher as $\alpha$ grows. This would mean that we'll have to classify every point perfectly to get $\alpha = 0$.

But ...

We know what this means right? **Overfitting**.

</details>

<details>

<summary>How can we handle this problem?</summary>

We can introduce a hyperparameter of the SVM that would control how much **regularization** we impose - this parameter is typically called **C**:

$$\alpha_i \in [0, C]$$

</details>

Interestingly enough this parameter **C** did already show up.

<details>

<summary>Which algorithm also had the parameter "C"?</summary>

`LogisticRegression`

</details>

So, `C` is the regularization parameter we'll employ in SVMs as well.

<details>

<summary>What was its meaning - as "C" increases what happens to the regularization strength?</summary>

It **decreases**! Notice, as we make the interval bigger, our model would start overfitting, since the adversary wants to maximize the values of the $\alpha$s.

![w05_ksvm7.png](assets/w05_ksvm7.png "w05_ksvm7.png")

</details>

<details>

<summary>Hmm, that is interesting - can we then use the values of the alphas to distinguish whether a sample is a support vector or not?</summary>

We can! The support vectors are the samples $i$ for which $\alpha_i$ is strictly positive!

> **Note:** In practice, the resulting alphas are very tiny, so we usually pick the support vectors based on a small positive values $\epsilon$, i.e. `support_ = np.where(alphas > 1e-5)[0]`.

</details>

Ok, great! So, how are we going to solve this? We already know that Gradient descent won't work as we have constraints?

We'll use a quadratic solver for this, more specifically [the package quadprog](https://github.com/quadprog/quadprog). We'll go back to how we're going to use it, but first let's talk about something of equal importance.

### The Kernel Trick

<details>

<summary>Why did we went though all of this trouble if we're going to, in the end, use a third-party package to solve it for us?</summary>

Because it shows us an interesting dependence between the loss function and the training samples.

</details>

<details>

<summary>What is the only information the loss function needs from our samples?</summary>

It **does *not*** depend on the individual samples, but ***only* on the *dot-product*** between every pair of them!

</details>

But the story does not end here - let's substitute $w$ in the prediction function and see what we get for an unknown sample $u$:

We take the initial form for predicting the positive class:

$$w^T u + b >= 0$$

And place $w$:

$$\sum_i \alpha_i y_i x_i^T u + b >= 0$$

<details>

<summary>What do we see?</summary>

**The decision rule also depends only on dot-products!** In this case, between the unknown vector and our support vectors.

</details>

This is a big deal because it let's us switch perspectives when we have linearly inseparable data. Let's see how the book "The Hundred-Page Machine Learning Book" presents this:

![w05_ksvm8.png](assets/w05_ksvm8.png "w05_ksvm8.png")
![w05_ksvm8_1.png](assets/w05_ksvm8_1.png "w05_ksvm8_1.png")

We can see that what initially seemed like inseparable data, becomes a very easy task in higher dimensions!

<details>

<summary>So, if the transformation is $F$, what do we need in order to classify a new point?</summary>

We need $F(x_i) \cdot F(u)$.

</details>

<details>

<summary>And what do we need in order to minimize our loss function?</summary>

We need $F(x_i) \cdot F(x_j)$.

</details>

<details>

<summary>Hmm - ok ... but what does that mean then?</summary>

If we define a function $K(m, n) = F(m) \cdot F(n)$, then $K$ is all we need!

</details>

We don't actually need to compute $F$ directly anywhere! We can use the kernel to get what we require! <- **This is referred to as the *kernel trick*!**

The function $K$ is called the *kernel* - it outputs an approximation of the dot-product in another space, **without knowing the transformation $F$**.

So, we don't need to know $F$, we only need to know $K$! This is the power of SVMs.

Great! What are some popular kernels? We can check them out [in sklearn's documentation](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py):

- Linear: $K(\mathbf{x}_1, \mathbf{x}_2) = \mathbf{x}_1^\top \mathbf{x}_2$.
- Polynomial: $K(\mathbf{x}_1, \mathbf{x}_2) = (\gamma \cdot \ \mathbf{x}_1^\top\mathbf{x}_2 + r)^d$.
- RBF: $K(\mathbf{x}_1, \mathbf{x}_2) = \exp\left(-\gamma \cdot {\|\mathbf{x}_1 - \mathbf{x}_2\|^2}\right)$:
  - The default kernel for Support Vector Machines in `scikit-learn`!
  - Prone to overfitting - watch out and use regularization!
    - When the parameter $\gamma$ is small, the Gaussians get shurk around the sample points and we get overfitting.
  - Measures similarity between two data points in ***infinite* dimensions**!
- Sigmoid: $K(\mathbf{x}_1, \mathbf{x}_2) = \tanh(\gamma \cdot \mathbf{x}_1^\top\mathbf{x}_2 + r)$

### Minimizing the objective

Let's go back now to the way we're optimizing this objective:

$$\min_{\alpha_i >= 0} L(\alpha) = \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^T x_j - \sum_i \alpha_i$$

Note, that now we can rewrite this as:

$$\min_{\alpha_i >= 0} L(\alpha) = \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j K(x_i, x_j) - \sum_i \alpha_i$$

We'll use the function `quadprog.solve_qp`. Here's its docstring:

```python
def solve_qp(double[:, :] G, double[:] a, double[:, :] C=None, double[:] b=None, int meq=0, factorized=False):
    """Solve a strictly convex quadratic program

    Minimize     1/2 x^T G x - a^T x
    Subject to   C.T x >= b

    This routine uses the the Goldfarb/Idnani dual algorithm [1].

    References
    ---------
    ... [1] D. Goldfarb and A. Idnani (1983). A numerically stable dual
        method for solving strictly convex quadratic programs.
        Mathematical Programming, 27, 1-33.

    Parameters
    ----------
    G : array, shape=(n, n)
        matrix appearing in the quadratic function to be minimized
    a : array, shape=(n,)
        vector appearing in the quadratic function to be minimized
    C : array, shape=(n, m)
        matrix defining the constraints under which we want to minimize the
        quadratic function
    b : array, shape=(m), default=None
        vector defining the constraints
    meq : int, default=0
        the first meq constraints are treated as equality constraints,
        all further as inequality constraints (defaults to 0).
    factorized : bool, default=False
        If True, then we are passing `R^{−1}` instead of the matrix G
        in the argument G, where `G = R^T R` and R is upper triangular.

    Returns
    -------
    x : array, shape=(n,)
        vector containing the solution of the quadratic programming problem.
    f : float
        the value of the quadratic function at the solution.
    xu : array, shape=(n,)
        vector containing the unconstrained minimizer of the quadratic function
    iterations : tuple
        2-tuple. the first component contains the number of iterations the
        algorithm needed, the second indicates how often constraints became
        inactive after becoming active first.
    lagrangian : array, shape=(m,)
        vector with the Lagragian at the solution.
    iact : array
        vector with the indices of the active constraints at the solution.
    """
```

We can see that it'll try to `Minimize     1/2 x^T G x - a^T x`.

> **Note:** Here the vector $x$ is not to our training samples. It represent the $\alpha$s that we're trying to find (see `Returns`, first entry in the tuple)!

We see that we have to set values to the following hyperparameters: `G`, `a`, `C`, `b`, `meq`, and `factorized`.

- What value would we give to $G$?
  - $G$ is known as the **Gram matrix** of the kernel: the matrix of pairwise kernel evaluations between all training samples.
  - It represent this part of the loss function: $\frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j K(x_i, x_j)$ ($\frac{1}{2}$ is just a constant, so we can ignore it).
  - Each entry tells us **how similar two samples are** according to the chosen kernel.
  - If we denote the kernel function as $K(x_i, x_j)$, then $G_{ij} = y_i y_j K(x_i, x_j)$.
- What value would we give to $a$?
  - This is the linear term in the loss function: $- \sum_i \alpha_i$.
  - Since the $x$s are the $\alpha_i$s, that means that $a$ is a vector with ones (`num_ones = num_samples`).
- What values would we give to $C$ and $b$?
  - $C$ represents the left side of the equations, while $b$ - the right side.
  - Our goal is to represent out two constraints $\sum_i \alpha_i y_i = 0$, $\alpha_i \in [0, C]$, so $C$ and $b$ will have three (vertical) parts:
    - constraining $\sum_i \alpha_i y_i = 0$:
      - the first in $C$ will be just the vector of mapped classes (to $-1$ and $1$);
      - the first in $b$ will be $0$, since the alphas should combine them in such a way that the sum is equal to $0$.
    - constraining $\alpha_i >= 0$:
      - the second in $C$ is an identity matrix with the size of the samples;
      - the second in $b$ is a vector of $0$s (`num_zeros = num_samples`) to represent the lower bound.
    - constraining $\alpha_i <= C$ via $-\alpha_i >= -C$:
      - the third in $C$ is the negated identity matrix with the size of the samples;
      - the third in $b$ is a vector of $-C$ (`num_neg_cs = num_samples`) to represent the upper bound.

<details>

<summary>What values would we give to "meq"?</summary>

- This tells `quadprog` how many of the constraints are equality constraints.
- We have $1$ such in the matrix $C$ and it is its first row with the mapped classes for each sample.

</details>

<details>

<summary>What value would we give to "factorized"?</summary>

We'll keep it at `False`, since we'll be passing the raw matrix `G`.

</details>

Once we solve the above optimization problem, we'll get the $\alpha$s (the first argument that `solve_qp` returns) and we can compute $b$:

$$b = \text{mean} \left[ y_i - \sum_j \alpha_j y_j K(x_j, x_i) \right]$$

only for samples $i$ where $0 < \alpha_i < C$

And our prediction function for the positive class becomes:

$$\sum_i \alpha_i y_i K(x_i, u) + b >= 0$$

### `SVC` in `scikit-learn`

This is the implementation of the `Kernel SVM` in `sklearn`.

```python
from sklearn import datasets, svm

wine = datasets.load_wine()

svm = svm.SVC()
svm.fit(wine.data, wine.target)
svm.score(wine.data, wine.target)
```

```console
0.7078651685393258
```

With default hyperparameters the accuracy is not particularly high, but it's possible to tune them to achieve `100%` training accuracy (by adjusting `gamma` and `C`). Such a classifier would be overfitting the training set, which is a risk we take when using more complex models.

### `LinearSVC` in `scikit-learn`

Apart from the `Kernel SVM` it is possible to train a **linear** classifier that maximizes the width of the street. It can be trained using stochastic gradient descent and thus is:

- creating the same boundary that we'd get by using a `Kernel SVM` with `kernel=linear`, but the optimization procedure is different.
- much faster than `Kernel SVM` to `fit` and `predict`;
- unable to work with the kernel trick since the decision boundary is linear.

In `scikit-learn` this flavor of SVM is implemented as the class `LinearSVC` (**linear** support vector classifier):

```python
from sklearn import datasets, svm

wine = datasets.load_wine()
print(type(wine))
print(dir(wine))
print(set(wine.target))

svm = svm.LinearSVC()
svm.fit(wine.data, wine.target)
svm.score(wine.data, wine.target)
```

```console
<class 'sklearn.utils._bunch.Bunch'>
['DESCR', 'data', 'feature_names', 'frame', 'target', 'target_names']
{np.int64(0), np.int64(1), np.int64(2)}
0.9887640449438202
```

We see that the `wine` dataset has more than `2` classes and the classifier handles them automatically using the one-vs-rest strategy (like `LogisticRegression`).

### Loss function diagrams

The linear SVM uses a different / new loss function - [the Hinge loss](https://en.wikipedia.org/wiki/Hinge_loss). In the soft margin SVM (a margin that allows for some misclassifications), the total loss combines the hinge loss with the regularization term:

$$
\min_{\mathbf{w}, b} \quad \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \max(0, 1 - y_i (\mathbf{w} \cdot \mathbf{x}_i + b))
$$

We want to draw loss functions, so let's set up a plot with `loss` on the vertical axis. On the horizontal axis we'll plot the `raw model output`.

Let's say that the training example is from class `+1`.Then, the right half represents correct predictions and the left half represents incorrect predictions.

![w07_loss_diagrams_1.png](assets/w07_loss_diagrams_1.png "w07_loss_diagrams_1.png")

Here's how the loss used by logistic regression looks like:

![w07_loss_logistic.png](assets/w07_loss_logistic.png "w07_loss_logistic.png")

<details>

<summary>What was the loss function for logistic regression?</summary>

Binary cross entropy / Log loss:

$\text{Log Loss} = \sum_{(x,y)\in D} -y\log(y') - (1 - y)\log(1 - y')$

</details>

And here is the [`hinge loss`](https://en.wikipedia.org/wiki/Hinge_loss) used by support vector machines in comparison:

![w07_loss_logistic_hinge.png](assets/w07_loss_logistic_hinge.png "w07_loss_logistic_hinge.png")

For an intended output $t = ±1$ and a classifier score $y$, the hinge loss of the prediction $y$ is defined as:

$${hidge(y)=\max(0,1-t\cdot y)} \text{,   where   } y = \mathbf{w}^\intercal \cdot \mathbf{x} - b$$

Note that as we move to the right, towards the zone of correct predictions, the loss goes down.

Which of the four loss functions makes sense for classification?
    ![w07_best_loss_fn.png](assets/w07_best_loss_fn.png "w07_best_loss_fn.png")

```text
A. (1)
B. (2)
C. (3)
D. (4)
```

<details>

<summary>Reveal answer:</summary>

B.

</details>

### Comparing logistic regression and SVM

Let's compare as a final step the two linear classifiers, logistic regression and linear SVMs.

| Logistic Regression                     | Linear Support Vector Machine                  |
| --------------------------------------- | ---------------------------------------------- |
| a linear classifier                     | a linear classifier                            |
| can utilize kernels, but is very slow   | can utilize kernels and fast                   |
| outputs easy-to-interpret probabilities | does not naturally output probabilities        |
| can be extended to multiclass           | can be extended to multiclass                  |
| all data points affect fit              | only "support vectors" affect fit              |
| has L1 and L2 regularization            | without extending, uses only L2 regularization |

Comparing the use in scikit-learn, we have:

| Logistic Regression                   | Linear Support Vector Machine                  |
| ------------------------------------- | ---------------------------------------------- |
| `linear_model.LogisticRegression`     | `svm.LinearSVC` and `svm.SVC(kernel='linear')` |
| `C` (inverse regularization strength) | `C` (inverse regularization strength)          |
| `penalty` (type of regularization)    | `kernel` (type of transformation)              |
| `multi_class` (type of multiclass)    | `gamma` (inverse RBF smoothness)               |

### [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)

This is a classifier that is a wrapper around logistic regression and linear SVMs. One "gotcha" is that the regularization hyperparameter is called `alpha`, instead of `C`, and bigger `alpha` means more regularization. Basically, `alpha = 1/C`.

```python
from sklearn.linear_model import SGDClassifier

# to switch between logistic regression and a linear SVM, one only has to set the "loss" hyperparameter

logreg = SGDClassifier(loss='log_loss')
logsvm = SGDClassifier(loss='hidge')
```

Which of the following is an advantage of SVMs over logistic regression?

```text
A. They naturally output meaningful probabilities.
B. They can be used with kernels.
C. They are computationally efficient with kernels.
D. They learn sigmoidal decision boundaries.
```

<details>

<summary>Reveal answer:</summary>

C. Having a limited number of support vectors makes kernel SVMs computationally efficient.

</details>

Which of the following is an advantage of logistic regression over SVMs?

```text
A. It naturally outputs meaningful probabilities.
B. It can be used with kernels.
C. It is computationally efficient with kernels.
D. It learns sigmoidal decision boundaries.
```

<details>

<summary>Reveal answer:</summary>

A.

</details>

# Week 06 - The Bias-Variance Tradeoff. Decision Trees

## Decision Trees

### Classification-And-Regression-Tree (`CART`)

#### Classification tree

##### What is a classification tree?

Given a labeled dataset, a classification tree learns a **sequence** of **if-else** questions about **individual features** in order to infer the labels.

In contrast to linear models, trees:

- capture ***naturally* non-linear relationships** between features and labels;
- don't require the features to be on the same scale through standardization/normalization.

Let's try to predict whether a tumor is malignant or benign in the [Wisconsin Breast Cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) using only `2` features.

![w07_dt_example1.png](assets/w07_dt_example1.png "w07_dt_example1.png")

When a classification tree is trained on this dataset, the tree learns a sequence of if-else questions.

Each question involves `1` feature and `1` **split-point**.

![w07_dt_example2.png](assets/w07_dt_example2.png "w07_dt_example2.png")

1. At the top, the tree asks whether the mean of the concave-points is <= `0.051`. If it is, the instance traverses the `True` branch; otherwise, it traverses the `False` branch.
2. The instance keeps traversing the internal branches until it reaches an end (a leaf node).
3. The label of the instance is then predicted to be that of the **prevailing class at that end**.

In scikit-learn the class for creating decision trees is called `DecisionTreeClassifier` and can be found in `sklearn.tree`. Here's how we could implement the above solution

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

dt = DecisionTreeClassifier(max_depth=2, random_state=1)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)
```

```console
0.90350877192982459
```

To understand the tree's predictions more concretely, let's see how it classifies instances in the feature-space.

![w07_dt_vs_logreg.png](assets/w07_dt_vs_logreg.png "w07_dt_vs_logreg.png")

- the left figure shows the decision regions of a logistic regression:
  - note how the boundary is a straight-line.
- the right figure shows the decision regions of a classification tree:
  - see how it produces rectangular decision regions in the feature space:
    - this happens because at each split made by the tree, only `1` feature is involved.

##### How does a classification tree learn?

Until now we know that:

- **Decision Tree:** a binary data structure consisting of a hierarchy of nodes;
- **Non-leaf Node:** question;
- **Leaf Node:** prediction.

![w07_dt_structure.png](assets/w07_dt_structure.png "w07_dt_structure.png")

The nodes of a classification tree are grown recursively. At each node, a tree asks a question involving one feature `f` and a split-point `sp`.

![w07_dt_structure_general.png](assets/w07_dt_structure_general.png "w07_dt_structure_general.png")

But how does it know which feature and which split-point to pick?

- It considers that every node contains an amount of uncertainty and aims to minimize that amount in the children nodes (make them more **pure**) with each split.

- The above is equivalent to saying that the tree maximizes the **information gain** it makes with every split.

Consider the case where a node with `N` samples is split into a left-node with `Nleft` samples and a right-node with `Nright` samples. The information gain (or the amount of uncertainty removed) for such split is given by this formula:

![w07_ig_formula.png](assets/w07_ig_formula.png "w07_ig_formula.png")

Here `I` is the amount of uncertainty and `IG` is information gain.

##### What criterion is used to measure the impurity of a node?

There are different criteria you can use among which are the **gini-index** and **entropy**.

**Gini Index Formula and Example:**

![w07_gini_formula.png](assets/w07_gini_formula.png "w07_gini_formula.png")

**Entropy Formula and Example:**

![w07_entropy_formula.png](assets/w07_entropy_formula.png "w07_entropy_formula.png")

Note that here when we're considering multiple splits, we take the weighted averages between the entropies for each split:

![w07_entropy_example.png](assets/w07_entropy_example.png "w07_entropy_example.png")

The default criteria in scikit-learn is `gini`, though we could also specify `entropy`. Often we would compare both in a grid search.

```python
dt_with_gini = DecisionTreeClassifier(criterion='gini') # default
dt_with_entropy = DecisionTreeClassifier(criterion='entropy')
```

<details>

<summary>
Which of the following is not true?

1. The existence of a node depends on the state of its predecessors.
2. The impurity of a node can be determined using different criteria such as entropy and the gini-index.
3. When the information gain resulting from splitting a node is null, the node is declared as a leaf.
4. When an internal node is split, the split is performed in such a way so that information gain is minimized.

Click to reveal answer.
</summary>

Answer: 4. It's quite the contrary - splitting an internal node always involves maximizing the information gain.

</details>

#### Regression tree

Below the [`automobile miles-per-gallon`](https://archive.ics.uci.edu/dataset/9/auto+mpg) dataset is shown (it's also present in our `DATA` folder as `auto.csv`).

![w07_mpg.png](assets/w07_mpg.png "w07_mpg.png")

It consists of `6` features corresponding to the characteristics of a car and a continuous target variable labeled `mpg` (miles-per-gallon). Our task is to predict the mpg consumption of a car given these features.

Let's try to do this by only using the displacement of a car - `displ`.

A 2D scatter plot shows that the mpg-consumption decreases **nonlinearly** with displacement.

![w07_mpg_scatter.png](assets/w07_mpg_scatter.png "w07_mpg_scatter.png")

When a regression tree is trained on a dataset, the **impurity** of a node is measured using the **mean-squared error** of the targets in that node.

![w07_regression_tree.png](assets/w07_regression_tree.png "w07_regression_tree.png")

This means that the regression tree tries to find the splits that produce leafs where in each leaf the target values are on average, the closest possible to the mean-value of the labels in that particular leaf.

As a new instance traverses the tree and reaches a certain leaf, its target-variable `y` is computed as the average of the target-variables contained in that leaf.

![w07_regression_tree_example.png](assets/w07_regression_tree_example.png "w07_regression_tree_example.png")

To highlight the importance of the flexibility of regression trees, take a look at this figure. The regression tree shows a greater flexibility and is able to capture the non-linearity, though not fully.

![w07_regression_tree_vs_lin_reg.png](assets/w07_regression_tree_vs_lin_reg.png "w07_regression_tree_vs_lin_reg.png")

### The Bias-Variance Tradeoff

In supervised learning, you make the assumption: $y = f(x), f$ is unknown.

$f$ shown in red is an unknown function that you want to determine. Real data, however, is always accompanied with randomness or noise like the blue points.

![w07_bv_tradeoff_example1.png](assets/w07_bv_tradeoff_example1.png "w07_bv_tradeoff_example1.png")

#### The goals of Supervised Learning

- find a model $\hat{f}$ that best approximates $f$: $\hat{f} \approx f$:
  - $\hat{f}$ can be any machine learning model: logistic regression, decision tree, neural network, etc.
- discard noise as much as possible;
- $\hat{f}$ should achieve a low predictive error on unseen data.

#### Difficulties in Approximating $f$

- **Overfitting:** $\hat{f}(x)$ fits the training set noise.

![w07_overfitting.png](assets/w07_overfitting.png "w07_overfitting.png")

- **Underfitting:** $\hat{f}$ is not flexible enough / complex enough to approximate $f$.

![w07_underfitting.png](assets/w07_underfitting.png "w07_underfitting.png")

#### Generalization Error

- **Generalization Error of $\hat{f}$**: Quantifies how well $\hat{f}$ generalizes on unseen data.
- It can be decomposed to:

$$\hat{f} = bias^2 + variance + irreducible\ error$$

- **Bias:** error term that quantifies how much on average the model fails to predict the true outcome ($\hat{f} \neq f$). Here is a model with high bias:

![w07_bias.png](assets/w07_bias.png "w07_bias.png")

- **Variance:** error term that quantifies how much $\hat{f}$ is inconsistent over different training sets (how overfit the model is). Here is a model with high variance:

![w07_variance.png](assets/w07_variance.png "w07_variance.png")

- **Irreducible error:** The error contribution of noise. There can never be a perfect model, so we regard this term as a small constant that is always present.

#### Model Complexity

- The easiest way in which we can control how well $\hat{f}$ approximates $f$ is by varying its (the model's) complexity.
- Examples: Maximum tree depth, Minimum samples per leaf, Number of features used, Number of neurons, etc.
- This diagram is known as the **Bias-Variance Tradeoff**: it shows how the best model complexity corresponds to the lowest generalization error.

![w07_bv_diagram.png](assets/w07_bv_diagram.png "w07_bv_diagram.png")

#### Bias-Variance Tradeoff: A Visual Explanation

Let's say that we want to create a model that predicts `2D` points.

The inner-most blue circle of the below diagrams represents perfect predictions with `0` error (in a certain threshold (the value of which does not matter)). The squares represent individual predictions.

The `x` and `y` axes represent the errors made for each coordinate - the center represents perfect predictions, so the errors are near `0`.

Then, we can visualize the bias-variance tradeoff with the following `4` diagrams:

![w07_bv_diagram2.png](assets/w07_bv_diagram2.png "w07_bv_diagram2.png")

#### Checkpoint

- Which of the following correctly describes the relationship between $\hat{f}$'s complexity and $\hat{f}$'s bias and variance terms?

```text
A. As the complexity of decreases, the bias term increases while the variance term decreases.
B. As the complexity of decreases, both the bias and the variance terms increase.
C. As the complexity of increases, the bias term increases while the variance term decreases.
D. As the complexity of increases, the bias term decreases while the variance term increases.
```

<details>

<summary>Click to reveal answer</summary>

Answer: D

</details>

- Visually diagnose whether a model is overfitting or underfitting the training set. Let's say you've trained two different models $A$ and $B$ on the `auto` dataset to predict the `mpg` consumption of a car using only the car's displacement (`displ`) as a feature. The following figure shows you scatterplots of `mpg` versus `displ` along with lines corresponding to the training set predictions of models $A$ and $B$ in red. Which of the following statements is true?

    ![w07_checkpoint.png](assets/w07_checkpoint.png "w07_checkpoint.png")

```text
A. (A) suffers from high bias and overfits the training set.
B. (A) suffers from high variance and underfits the training set.
C. (B) suffers from high bias and underfits the training set.
D. (B) suffers from high variance and underfits the training set.
```

<details>

<summary>Click to reveal answer</summary>

Answer: C. Model B is not able to capture the nonlinear dependence of `mpg` on `displ`.

</details>

### Train-test split revisited

#### Estimating the Generalization Error

- How do we estimate the generalization error of a model?
- Cannot be done directly because:
  - $f$ is unknown (if it was known, we would've just coded the formula);
  - usually there's only one dataset;
  - noise is unpredictable.
- Solution: split the data into training and testing sets:
  - fit $\hat{f}$ to the training set and evaluate the its error on the **unseen** test set;
  - the generalization error of $\hat{f}$ $\approx$ test set error of $\hat{f}$.
  - there's a problem with this approach, though: the test set should not be used until we're confident about $\hat{f}$'s performance.
    - also, we can't evaluate $\hat{f}$ on the training set as that would give a biased estimate ($\hat{f}$ has already seen all training points).

<details>

<summary>What is the solution?</summary>

K-Fold cross validation!

![w07_kfold_recap_1.png](assets/w07_kfold_recap_1.png "w07_kfold_recap_1.png")

The error is then calculated as the mean of the cross-validation results:

![w07_kfold_recap_2.png](assets/w07_kfold_recap_2.png "w07_kfold_recap_2.png")

</details>

#### Diagnose Variance Problems

- If CV error of $\hat{f}$ > training set error of $\hat{f}$: $\hat{f}$ suffers from **high variance**;
- $\hat{f}$ is said to have **overfit** the training set. To remedy overfitting:
  - decrease model complexity;
    - decrease max tree depth, increase min samples per leaf, decrease number of neurons.
  - gather more data.

#### Diagnose Bias Problems

- If CV error of $\hat{f} \approx$ training set error of $\hat{f}$ and this error is much greater than the disired error: $\hat{f}$ suffers from **high bias**;
- $\hat{f}$ is said to have **underfit** the training set. To remedy underfitting:
  - increase model complexity;
    - increase max tree depth, decrease min samples per leaf, increase number of neurons, increase number of layers.
  - gather more relevant features;
  - feature engineering.

- What do we deduct from the below outputs - overfitting or underfitting?

    ![w07_kfold_checkpoint.png](assets/w07_kfold_checkpoint.png "w07_kfold_checkpoint.png")

<details>

<summary>Click to reveal answer</summary>

Answer: overfitting. The cross-validation error is higher than the training error.

</details>

### Ensemble Learning

#### Advantages of CARTs

- Simple to understand;
- Simple to interpret;
- Easy to use;
- Flexibility: ability to describe non-linear dependecies;
- Preprocessing: no need to standardize or normalize features.

#### Limitations of CARTs

- Classification: can only produce orthogonal decision boundaries;
- Sensitive to small variations in the training set;
- High variance: unconstrained CARTs easily overfit the training set;
- We can address these limitations by utilizing the **Ensemble Learning** technique.

#### What is Ensemble Learning?

- Train different models on the same dataset;
- Let each model make its predictions;
- Create a meta-model that aggregates the predictions of individual models;
- Output the final prediction. Using this technique, we get more robust results that are less prone to errors;
- Usually the best results are obtained when the used models are skillful in different ways:
  - this can be achieved easily if different types of models are used (rather than a variation of the same model).

![w07_ensemble.png](assets/w07_ensemble.png "w07_ensemble.png")

#### The [`Voting Classifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier)

Let's say we have a binary classification task.

- We create $N$ classifiers and get their predictions for a single observation: $P_0, P_1, \dots, P_{n-1}$ with $P_i = 0$ or $1$.
- Those predictions get passed to a meta-model. In the case of the `Voting Classifier`, this model is simply a majority vote on the predictions.
  - If $N$ is even and the predictions get event, a class is chosen at random.
  - Therefore, aviod setting $N$ to an even number.

![w07_ensemble_voting_clf.png](assets/w07_ensemble_voting_clf.png "w07_ensemble_voting_clf.png")

# Week 07 - Bagging. Boosting

## Bagging

We finished off last week by talking about how we can ensemble multiple models to make a prediction.

<details>

<summary>What are the disadvantages of ensemble methods?</summary>

- Trained on the same training set.
- Use different algorithms.
- Do not guarantee better performance (as you'll see in the tasks).

</details>

<details>

<summary>What does bagging stand for?</summary>

Shorthand for **B**ootstrap **agg**regation.

</details>

<details>

<summary>Do you know what bagging means?</summary>

- A model creation technique that outputs an ensemble meta-estimator that fits base classifiers **of 1 model type** each on random subsets of the original dataset and then aggregates their individual predictions (either by voting or by averaging) to form a final prediction.
  - Training dataset for each model is formed by the `Bootstrap` technique.
  - Proven to reduce variance of individual models in the ensemble.

</details>

### Bootstrap

<details>

<summary>What is a bootstrap sample?</summary>

A bootstrap sample is a set of randomly chosen observations from the original dataset.

</details>

<details>

<summary>Are samples drawn with or without replacement?</summary>

The samples are drawn with replacement.

</details>

<details>

<summary>How many samples are drawn from the original dataset?</summary>

The size of the bootstrap sample is the same as the same size of the original dataset.

</details>

<details>

<summary>If we have a dataset with 3 observations could you visualize the creation of 3 bootstrap samples?</summary>

![w07_bootstrap.png](assets/w07_bootstrap.png "w07_bootstrap.png")

</details>

### Bagging: Training

<details>

<summary>Do you know what happens during the training phase of a bagging algorithm?</summary>

In the training phase, bagging consists of:

1. Drawing `N` bootstrap samples from the training set (`N` is a hyperparameter).
2. Using each of them to train `N` models that use the same algorithm.

![w07_bagging_training.png](assets/w07_bagging_training.png "w07_bagging_training.png")

</details>

### Bagging: Prediction

<details>

<summary>Can you intuit how prediction might work for a bagging algorithm?</summary>

When a new instance is fed to the different models forming the bagging ensemble, each model outputs its prediction. The meta model collects these predictions and outputs a final prediction depending on the nature of the problem.

![w07_bagging_prediction.png](assets/w07_bagging_prediction.png "w07_bagging_prediction.png")

</details>

<details>

<summary>How would prediction work for a classification task?</summary>

- If base estimators have a method `predict_proba`, then we return the class with the highest mean predicted probability from each of the estimators.
- If base estimators do not have a method `predict_proba`, then we aggregates the predictions by majorty voting.
- [`BaggingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) in scikit-learn.

</details>

<details>

<summary>How would prediction work for a regression task?</summary>

- Aggregates predictions through averaging.
- [`BaggingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html) in scikit-learn.

</details>

<details>

<summary>Open one of the bagging algorithms above - do we pass the class for the parameter "estimator" or an instance of it?</summary>

The model you pass as an `estimator` must already be created / instanciated, but not fitted.

</details>

<details>

<summary>Do bagging algorithms work only for decision trees or can we use any algorithm in bagging?</summary>

Any model will do.

</details>

### Bagging Reduces Variance

Let's see why bagging reduces variance and why decision trees are typically the default estimators that are bagged.

<details>

<summary>What did we see last time - do decision trees suffer from high variance?</summary>

Yep - they are one of the fastest algorithms to overfit the training set.

</details>

<details>

<summary>And do they suffer from high bias?</summary>

Actually, no - they are an **unbiased** estimator. They do not have any assumptions about the training data.

</details>

<details>

<summary>Let's give a counter-example to this - what is the main assumption of linear models (that makes them biased)?</summary>

That there is a linear relationship (correlation) between at least one of the features and the target variable.

</details>

Suppose we bag $M$ decision trees. Each of them has variance $\sigma^2$ and bias $b$.

<details>

<summary>How much is the bias?</summary>

Since we have a decision tree, $b = 0$.

</details>

<details>

<summary>Are they independent?</summary>

We are using the same data so they are not completely independent and do have a small amount of correlation.

</details>

Let's say that each estimator $x$ has variance $\hat{f}_m(x)$ (some large number).

<details>

<summary>How much would the variance of the bagged model be?</summary>

If the correlation between any two estimators is $\rho$, then:

$$
\text{Var}\left( \hat{f}_{\text{bag}}(x) \right) = \rho \sigma^2 + \frac{(1-\rho)\sigma^2}{M}
$$

*   When $\rho = 0$ (perfect independence), the variance reduces to ( $\sigma^2 / M$ ).
*   When $\rho = 1$ (perfect correlation), the variance stays at ( $\sigma^2$ ) — no benefit.
*   Bagging works best when base estimators are **unstable** and have low correlation after bootstrapping. If this is the case, the variance of the bagged models will be lower than the variance of each individual model and would decrease as we add more estimators.

</details>

And this is why decision trees are ideal for bagging:

*   **High Variance Models**: They are very sensitive to small changes in data (high variance). Bagging reduces this variance dramatically.
*   **Low Bias**: They can fit complex patterns, so bias is low. Bagging doesn’t fix bias, so starting with low bias is good.
*   **Bootstrap Sampling Reduces Correlation**: Different bootstrap samples lead to diverse trees, lowering correlation.

### The `oob_score_`

During bagging some instances may be sampled several times for one model, but others may not be sampled at all. In fact, on average only $67%$ of the initial samples appear in the bootstrap sample!

<details>

<summary>How can we make use of the observations that are not sampled?</summary>

We can use them to evaluate the model. This is known as the out-of-bag (`OOB`) evaluation. You can think of this as an analogy (**not the same!**) to cross validation / train-test splitting:

![w07_oob_evaluation.png](assets/w07_oob_evaluation.png "w07_oob_evaluation.png")

In sklearn, we can obtain the out of bag scores of each model by doing these two steps:

1. Set `oob_score` to `True` when instanciating the aggregator.
2. Train the model using the training set.
3. Access the `OOB` scores using the `oob_score_` attribute of the trained model.

> **Note:** When `oob_score_` is set to a boolean it returns the **accuracy** on the OOB instances. You'd have to pass a call to the metric `f1` if you wish to visualize this.

</details>

### Further Diversity with `Random Forests`

- `Random Forests` is another ensemble learning method.
- Its base estimator is a `Decision Tree`.
- Each estimator is trained on a different bootstrap sample with size = size of original dataset.

<details>

<summary>So what is the difference between bagging algorithms and random forests?</summary>

They introduce further randomization in the training of individual trees by now only sampling rows (bootstrapping observations), but also sampling columns ("bootstrapping" (with a twist - without replacement) features):

$d$ features are sampled **for each model**, but ***without*** replacement ($d < \text{total number of features}$).

![w07_random_forest.png](assets/w07_random_forest.png "w07_random_forest.png")

</details>

<details>

<summary>How are predictions made when we have classification?</summary>

- A vote by the trees in the forest, weighted by their probability estimates. That is, the predicted class is the one with highest mean probability estimate across the trees.
- [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) in scikit-learn.

</details>

<details>

<summary>How are predictions made when we have regression?</summary>

- The predicted regression target of an input sample is computed as the mean predicted regression targets of the trees in the forest.
- [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) in scikit-learn.

</details>

> **Note:** Similarly to bagging, not only trees can be used, but the model you pass as an `estimator` must already be created / instanciated.

### Feature Importance with Trees and Forests

Tree-based methods enable measuring the importance of each feature.

In `sklearn`:

- how many times the tree nodes use a particular feature (weighted average) to reduce impurity.
- accessed using the attribute `feature_importance_`.

<details>

<summary>What other algorithm could we use to determine feature importances?</summary>

Lasso Regression!

</details>

## Boosting

<details>

<summary>What is the essence of boosting?</summary>

It is an ensemble method combining several **weak learners** to form a strong learner.

</details>

<details>

<summary>What is your intuition about weak learners?</summary>

A model doing slightly better than random guessing.

</details>

<details>

<summary>In the context of decision trees what would be an example of a weak learners?</summary>

A decision stump (CART whose maximum depth is $1$).

</details>

The general form of boosting algorithms is:

1. Train an ensemble of predictors sequentially.
2. Each predictor tries to correct its predecessor.

<details>

<summary>This second point is rather abstract - what does it mean in your opinion to "try to correct predecessor errors"?</summary>

There're several possibilities:

1. Place weights on the samples and use the [weighted gini impurity](https://stats.stackexchange.com/questions/68940/how-is-the-weighted-gini-criterion-defined) measure to forcefully increase the impurity when hard samples are not properly split, i.e. force the stumps to find better features that split hard / heavy samples.
2. Place weights on the samples and for each new stump duplicate the hard samples according to their weight. **<- this is what we'll use in our implementation**
3. Make the new weak learner predict the mistakes of its direct predecessor. <- this the approach taken by gradient boosting algorithms

</details>

- Most popular boosting methods:
  - AdaBoost;
  - Gradient Boosting family:
    - Vanilla Gradient Boosting;
    - XGBoost;
    - CatBoost; **<- always try this algorithm out. It has built-in handling for categorical features.**
    - LightGBM.

### Adaboost

- Shorthand for **Ada**ptive **Boost**ing.
- Each predictor pays more attention to the instances wrongly predicted by its predecessor.
  - Achieved by changing the weights of training instances, thus increasing the value of the loss if they get uncorrectly prediced again.
- Each predictor is assigned a coefficient $\alpha$ that corresponds to the **amount of say** the predictor has in determining the final vote.
  - $\alpha$ depends on the predictor's training error.

The training procedure is as follows:

1. Each of the samples is assigned a weight $1 / N, \text{N = number of samples}$.
2. A stump is trained on the data.
3. Get the samples that were wrongly classfied in each of the leafs.
4. Calculate the total error of the stump: $\text{Total Error} = \sum w_{\text{misclassified}}$.
5. Calculate the amount of say of the stump: $\text{Amount of Say} = \frac{1}{2} \log_e{\left[(1 - \text{Total Error}) / \text{Total Error} \right]}$.
6. Update the sample weights of the correctly classified samples: $w_{\text{misclassified}} = w_{\text{misclassified}} \exp{\left[ \text{Amount of Say} \right]}$.
7. Update the sample weights of the incorrectly classified samples: $w_{\text{not misclassified}} = w_{\text{not misclassified}} \exp{\left[ \text{-Amount of Say} \right]}$.
8. Recreate $X$ using these new weights by sampling with weights and replacement. Thus, we duplicate the misclassified entries.
9. Reset the weights to be uniform.

![w07_adaboost.png](assets/w07_adaboost.png "w07_adaboost.png")

### Learning Rate

Learning rate: $0 < \eta \leq 1$

![w07_adaboost_lr.png](assets/w07_adaboost_lr.png "w07_adaboost_lr.png")

The learning rate - $\eta$ (eta), is used to shrink the coefficient $\alpha$ of a trained predictor. It's important to note that there's a trade-off between $\eta$ and the number of estimators. **A smaller value should be compensated by a greater number of estimators.**

### AdaBoost: Prediction

**Classification:**

- Weighted mean prediction of the classifiers in the ensemble. Weighted by the amount of say.
- In scikit-learn [`AdaBoostClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html).

**Regression:**

- Weighted median prediction of the regressors in the ensemble. Weighted by the amount of say.
- In scikit-learn [`AdaBoostRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html).

### Gradient Boosted Trees

- Each predictor is trained using its predecessor's residual errors as **labels**.
- The base learner is a CART.
- Result: sequential correction of predecessor's errors.

### Gradient Boosted Trees for Regression: Training

Each predictor is trained to predict the last one's residuals.

![w07_gradient_boosted_trees_regression.png](assets/w07_gradient_boosted_trees_regression.png "w07_gradient_boosted_trees_regression.png")

For classification, the residuals are calculated as the difference between the class probabilities.

### Shrinkage

- The prediction of each tree in the ensemble is shrinked after it is multiplied by a learning rate $\eta$.
- Similarly to `AdaBoost`, there's a trade-off between eta and the number of estimators:
  - less learning rate => a lot of estimators

![w07_gradient_boosted_trees_eta.png](assets/w07_gradient_boosted_trees_eta.png "w07_gradient_boosted_trees_eta.png")

### Gradient Boosted Trees for Regression: Prediction

- Regression:
  - $y_{pred} = y_1 + \eta r_1 + \ldots + \eta r_N$
  - in sklearn: [`GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).

- Classification:
  - in sklearn: [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

### Gradient Boosting: Cons

Each CART is trained to find the best split points and features. This produces two problems:

- this is an exhaustive search procedure;
- may lead to CARTs using the same split points and maybe the same features.

### Stochastic Gradient Boosting

- We can use the classes `GradientBoostingRegressor` and `GradientBoostingClassifier`.
  - Making them stochastic is done by changing the values for the parameters `subsample` and `max_features`.
    - Each tree is trained on a random subset of rows and columns of the training data.
- The sampled instances are sampled without replacement.
- When choosing split points the features are also sampled without replacement.
- Benefits:
  - Further ensemble diversity.
  - Added bias to the ensemble of trees, thus ability to reduce overfitting.

![w07_stochastic_gradient_boosting.png](assets/w07_stochastic_gradient_boosting.png "w07_stochastic_gradient_boosting.png")

# Week 08 - Unsupervised Learning

## The added value of unsupervised learning

<details>

<summary>What is unsupervised learning?</summary>

It represents a class of machine learning algorithms that find patterns in unlabelled data.

</details>

<details>

<summary>Give at least two examples for unsupervised learning?</summary>

- *clustering* customers by their purchases;
- *reducing the dimensionality* of the customers' data using their purchase patterns.

</details>

<details>

<summary>What is the main difference between supervised learning and unsupervised learning?</summary>

- Supervised learning finds patterns that best help in solving a prediction task.
  - Example: classify tumors as benign or cancerous (given a set of *labels*).
- Unsupervised learning finds patterns in data, but *without* a specific prediction task in mind.

</details>

Let's say we are given a very large dataset: ~10,000,000 observations. The task is to create a classification model that assigns new data into $4$ classes. But there is a twist - because labelling each observation is very costly and requires expert attention, only $20$ samples in the dataset are labelled and the class distribution in them is very unbalanced:

```text
class 1: 1 sample
class 2: 1 sample
class 3: 1 sample
class 4: 17 samples
```

To create a model, we would like to use the full dataset and have $100\%$ of the labels.

<details>

<summary>How can we label the remaining samples?</summary>

We could use supervised learning:

1. Train a model on the $20$ samples.
2. Predict for the remaining ones.
3. Create a new model using the full dataset.

</details>

We tried out this approach, but every model overfit the data. Essentially, we don't have enough data and the class distribution does not allow us to create a model that can label the remaining data in an unbiased way.

<details>

<summary>What can we do in this situation?</summary>

We should use unsupervised learning:

1. Since we already know that the possibilities for each sample are only $4$, we can train a clustering algorithm that would split the dataset into $4$ groups/clusters.
2. We have several labelled samples and we could use them to evaluate the quality of the clustering using them. It wouldn't be a perfect measure, but at least it'll give us a number.
3. After we do the clustering, we'll have labels for all samples and we could use them to create our classification algorithm.

</details>

Let's elaborate more on the second point.

<details>

<summary>How can we evaluate the quality of the cluster using labelled data?</summary>

We can check the correspondence between assigned labels and actual labels: do the clusters correspond to the actual species?

We do that and get this distribution (the numbers here are exemplary):

![w08_kmeans_quality1.png](assets/w08_kmeans_quality1.png "w08_kmeans_quality1.png")

</details>

<details>

<summary>How can we interpret what we see?</summary>

We see that cluster $1$ corresponds perfectly with the species `setosa`. On the other hand, while cluster $0$ contains mainly `virginica` samples, there are also some `virginica` samples in cluster $2$.

- The table displaying the unique values of `labels` vs the unique values of `species` is called a **`cross-tabulation`**.
- We create it with [`pandas.crosstab`](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html) with the following code:

```python
import pandas as pd
df = pd.DataFrame({'labels': labels, 'species': species})
df
```

![w08_kmeans_quality2.png](assets/w08_kmeans_quality2.png "w08_kmeans_quality2.png")

```python
pd.crosstab(df['labels'], df['species'])
```

![w08_kmeans_quality3.png](assets/w08_kmeans_quality3.png "w08_kmeans_quality3.png")

</details>

Having species information in the dataset is very helpful, but it gives us a very limited view into the performance of the algorithm.

<details>

<summary>How can we evaluate a clustering algorithm on an unlabelled set?</summary>

**Inertia** or *within-cluster sum-of-squares*!

We have $n$ samples $x \in X$ and we want to distribute them into $C$ disjoint clusters, each cluster described by the mean $\mu_j$ of the samples in the cluster. The means are commonly called the cluster `centroids`; note that they are not, in general, points from $X$, although they live in the same space. The inertia, thus, is the following sum:

$$\text{Inertia} = \sum_{i=0}^{n}\min_{\mu_j \in C}(||x_i - \mu_j||^2)$$

- Inertia measures how spread out the clusters are.
- It uses the distance of each sample to the centroid of its cluster.
- Available after `fit()` as the attribute `inertia_`.

</details>

<details>

<summary>What happens to the performance of the model as the inertial increases/decreases; what is their relationship?</summary>

Lower inertial = better model.

</details>

<details>

<summary>What happens to the inertia as the number of clusters increases/decreases; what is their relationship?</summary>

More clusters means lower inertia.

</details>

<details>

<summary>What is the problem with this?</summary>

A high amount of clusters may not correspond to reality.

</details>

We have to be very careful as the simpler algorithms require us to specify it, i.e. it's a hyperparameter.

So, we have to answer the question: how many should we choose?

The short answer is to always experiment, try out several possibilities and visualize the results.

<details>

<summary>How can we decide on the number of clusters for one-dimensional and two-dimensional data?</summary>

We create a scatter plot:

![w08_points_2d.png](assets/w08_points_2d.png "w08_points_2d.png")

</details>

<details>

<summary>What if our data has three or more dimensions?</summary>

1. Fit various number of clusters and plot their values against the `inertia`.
2. Choose an `elbow` point - a point after which there is little to no added value.

![w08_elbow.png](assets/w08_elbow.png "w08_elbow.png")

</details>

<details>

<summary>What would we choose for the above diagram?</summary>

$3$ would most likely be the optimal number of clusters, though again - it'd be good if we could compare the concrete numbers between $2$, $3$ and $4$ clusters.

</details>

## The [`K-Means Clustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) algorithm

<details>

<summary>If you knew that KMeans works analogously to KNN, what would be your intuition for the steps it executes?</summary>

1. Chooses `n_clusters` random samples.
2. Labels them as centroids.
3. Assigns each sample to the closest centroid.
4. Recomputes the centroids to be the mean of all samples in each cluster.
5. Repeats steps 3 and 4 until the new centroids equal the old ones.

</details>

Here is a one-dimensional dataset:

![w08_kmeans_walk_through.png](assets/w08_kmeans_walk_through.png "w08_kmeans_walk_through.png")

Visualize the iterations KMeans makes.

<details>

<summary>What do you get?</summary>

For the steps in-between, please sync with Simo. Here is the final result:

![w08_kmeans_walk_through_final.png](assets/w08_kmeans_walk_through_final.png "w08_kmeans_walk_through_final.png")

</details>

Here is a two-dimensional dataset:

![w08_kmeans_2d_init.png](assets/w08_kmeans_2d_init.png "w08_kmeans_2d_init.png")

<details>

<summary>What would we get now?</summary>

![w08_kmeans_2d.gif](assets/w08_kmeans_2d.gif "w08_kmeans_2d.gif")

</details>

## DBScan and HDBScan

Apart from KMeans, other more powerful algorithms that deserve mentioning are:

- [DBScan](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html):
  - Views clusters as areas of high density separated by areas of low density.
  - The clusters found by DBSCAN can be any shape, as opposed to k-means which assumes that clusters are convex shaped.
  - The central component to the DBSCAN is the concept of **core samples**: samples that are in areas of high density.
  - A cluster is therefore a set of core samples, each close to each other (measured by some distance measure) and a set of non-core samples that are close to a core sample (but are not themselves core samples).
  - There are two parameters to the algorithm, `min_samples` and `eps`, which define formally what we mean when we say dense.
    - Higher `min_samples` or lower `eps` indicate higher density necessary to form a cluster.
- [HDBScan](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html):
  - Assumes that the clustering criterion (i.e. density requirement) is globally homogeneous.
  - Performs DBSCAN over varying `epsilon` values and integrates the result to find a clustering that gives the best stability over epsilon. This allows HDBSCAN to find clusters of **varying densities** (unlike DBSCAN), and be more robust to parameter selection.

## Scaling features for better clusters

Let's look at the [Piedmont wines dataset](https://archive.ics.uci.edu/dataset/109/wine):

- 178 samples from 3 varieties of red wine: Barolo, Grignolino and Barbera.
- Features measure:
  - Chemical composition e.g. alchohol content.
  - Visual properties like "color intensity".

If we run `KMeans` with three clusters, we would get clusters that don't correspond well with the wine varieties.

![w08_wine.png](assets/w08_wine.png "w08_wine.png")

This is how the clusters look like for two of the features:

![w08_wine_variances.png](assets/w08_wine_variances.png "w08_wine_variances.png")

<details>

<summary>What might be the cause for the low performance?</summary>

This is because the wine features have very different **scales**/**variances**:

- `od280` takes values between $1$ and $4$;
- `malic_acid` takes values between $0.2$ and $5.8$.

</details>

The model did its best, but the data points themselves are just not that different from each other.

We analyze the variance of the features is analyzed during the data audit:

![w08_wine_proline.png](assets/w08_wine_proline.png "w08_wine_proline.png")

We have to decrease the variance of the features and hope that we'll get points that are more grouped together.

<details>

<summary>How can we do that?</summary>

- In `KMeans` clustering, the variance of a feature is directly proportional to its influence on the clustering algorithm.
- [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) transforms each feature to have a mean of `0` and a variance of `1`, giving each feature a chance.

![w08_ss_viz.png](assets/w08_ss_viz.png "w08_ss_viz.png")

- Result: Much better grouping of points!
  - **In general, every algorithm that depends on distance calculations benefits from having the input feature values scaled!**

![w08_wine_proline_scaled.png](assets/w08_wine_proline_scaled.png "w08_wine_proline_scaled.png")

Effectively, we get a much better model:

Before scaling:

![w08_wine.png](assets/w08_wine.png "w08_wine.png")

After scaling:

![w08_wine_scaled.png](assets/w08_wine_scaled.png "w08_wine_scaled.png")

</details>

`StandardScaler` is one of several possible "data preprocessing" steps.

<details>

<summary>What are some alternatives?</summary>

- `MaxAbsScaler`: The values are mapped across several ranges depending on whether negative OR positive values are present:
  - if only positive values are present, the range is $[0, 1]$;
  - if only negative values are present, the range is $[-1, 0]$;
  - if both negative and positive values are present, the range is $[-1, 1]$
  - MaxAbsScaler suffers from the presence of large outliers.

![w08_maxabs_viz.png](assets/w08_maxabs_viz.png "w08_maxabs_viz.png")

- `Normalizer`: Rescales the vector for **each sample** to have unit norm, independently of the distribution of the samples.

![w08_norm_viz.png](assets/w08_norm_viz.png "w08_norm_viz.png")

See other comparisons in [sklearn's preprocessing section](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#standardscaler).

</details>

It is up to us to experiment with different scalers/normalizers and choose the one that works best for our task.

## Visualization communicates insight

- A huge part of the work of a data scientist / data analyst is communication insights to other people, esp. non-technical, stakeholder-like people.
- We'll now discuss two unsupervised learning techniques for visualization:
  - `Hierarchical clustering`;
  - `t-SNE`: Creates a 2D map of a dataset.

### `Agglomerative` hierarchical clustering

In `sklearn` this is implemented via the class [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html).

The [Eurovision scoring dataset](https://eurovision.tv/history/full-split-results):

- 2D array of scores.
- Rows are countries, columns are songs.

![w08_evsn.png](assets/w08_evsn.png "w08_evsn.png")

- If we run the data through a hierarchical clustering of voting countries, we get a tree-like diagram, called a `dendrogram`.
- The dendrogram groups countries into larger and larger clusters. Many of those clusters are recognizable as containing countries that:
  - are close to one another geographically;
  - have cultural ties;
  - have political ties;
  - belong to the same language group.

![w08_example.png](assets/w08_example.png "w08_example.png")

- Dendrograms are read from the bottom up.
- Vertical lines represent clusters.

<details>

<summary>Can you intuit how agglomerative hierarchical clustering builds clusters?</summary>

1. Initially, every country is in its own separate cluster.
2. The distances between all pairs of clusters are computed. The distance metric is a hyperparameter.
3. The clusters with distance less than `distance_threshold` (hyperparameter, if not specified, the minimum distance is used) are merged.
4. Steps 2 and 3 are repeated until all countries are in a single cluster.

</details>

In `sklearn` there is no built-in function to plot dendrograms. Instead, we can use the functions from the [`scipy.cluster.hierarchy`](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) module to perform agglomerative clustering:

- `linkage`: to create the clusters;
- `dendrogram`: to create the dendrogram plot.

```python
from scipy.cluster import hierarchy
mergings = hierarchy.linkage(samples, method='complete')
hierarchy.dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()
```

- `method` is the distance metric. For example, `method='complete'` means: distance between clusters is the maximum distance between their samples. In other words, the distance between two clusters is defined by the distance between their farthest points.
- Different linkage methods produce different hierarchical clusterings.

![w08_linkage_options.png](assets/w08_linkage_options.png "w08_linkage_options.png")

<details>

<summary>If there are 5 data samples, how many merge operations will occur in agglomerative hierarchical clustering?</summary>

$4$ merges. Look back in the dendrogram from Eurovision to see the pattern.

</details>

<details>

<summary>How do we extract the clusters a dendrogram?</summary>

- Using its **height**!
- E.g. at height $15$:
  - Bulgaria, Cyprus, Greece are one cluster;
  - Armenia in a cluster on its own;
  - Russia and Moldova are another.

![w08_cluster_height.png](assets/w08_cluster_height.png "w08_cluster_height.png")

</details>

<details>

<summary>So, what is the meaning of the height?</summary>

- Height on dendrogram = distance between merging clusters.
- E.g. clusters with only Cyprus and Greece had distance approx. $6$.
- When this cluster was merged with the cluster containing Bulgaria, the distance between them (the merged cluster and the cluster of Bulgaria) was $12$.

![w08_cluster_distance.png](assets/w08_cluster_distance.png "w08_cluster_distance.png")

</details>

Programmatically, we can extract cluster labels with the function `fcluster()`. It returns a NumPy array of cluster labels.

```python
from scipy.cluster import hierarchy
mergings = hierarchy.linkage(samples, method='complete')
hierarchy.fcluster(mergings, 15, criterion='distance')
labels
```

```console
[ 9  8 11 20  2  1 17 14 ... ]
```

Consider the three clusters in the diagram. Which of the following statements are true?

![w08_checkpoint1.png](assets/w08_checkpoint1.png "w08_checkpoint1.png")

**A.** In single linkage, `Cluster 3` is the closest cluster to `Cluster 2`.

**B.** In complete linkage, `Cluster 1` is the closest cluster to `Cluster 2`.

1. Neither A, nor B.
2. A only.
3. B only.
4. Both A and B.

<details>

<summary>Reveal answer</summary>

Answer: `4. Both A and B.`

</details>

Displayed is the dendrogram for the hierarchical clustering of the grain samples. If the hierarchical clustering were stopped at height $6$ on the dendrogram, how many clusters would there be?

![w08_checkpoint2.png](assets/w08_checkpoint2.png "w08_checkpoint2.png")

<details>

<summary>Reveal answer</summary>

Answer: 3 - if we draw a horizontal line at `6`, we would see them.

</details>

### t-distributed stochastic neighbor embedding (`t-SNE`)

Open the documentation for [t-SNE in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

<details>

<summary>What is it used for?</summary>

- Maps samples to `2D` or `3D` space (essentially, reducing their dimensionality).
- Map approximately preserves nearness of samples.
- Great for inspecting high-dimensional datasets.
- It is highly recommended to use another dimensionality reduction method (e.g. `PCA` for dense data or `TruncatedSVD` for sparse data) to reduce the number of dimensions to a reasonable amount (e.g. $50$) if the number of features is very high. This will suppress some noise and speed up the computation of pairwise distances between samples.

</details>

Let's go back to the Iris dataset: it has $4$ measurements, so samples are $4$-dimensional. Let's use `t-SNE` to map the samples to `2D` space. We won't give it information about the species of each sample.

```python
from sklearn import manifold
model = manifold.TSNE(learning_rate=100)
transformed = model.fit_transform(samples)
xs = transformed[:, 0]
ys = transformed[:, 1]
plt.scatter(xs, ys, c=species)
plt.show()
```

![w08_tsne_ex1.png](assets/w08_tsne_ex1.png "w08_tsne_ex1.png")

<details>

<summary>How can we interpret the result?</summary>

We can see that the species were kept mostly separate.

We also learn that there are two iris species - `versicolor` and `virginica`, whose samples are close together in space, meaning that it's **hard to distinguish one from the other**.

</details>

<details>

<summary>Continue to read through the documentation - what is the meaning of the parameter "learning_rate"?</summary>

- It's the step size during the gradient descent optimization of its cost function.
- The learning rate for `t-SNE` is usually in the range `[10.0, 1000.0]`.
  - If too high, the data may look like a `ball` with any point approximately equidistant from its nearest neighbors.
  - If too low, most points may look compressed in a dense cloud with few outliers.
  - Basically, if the value is not ok, we'll get points that are bunched together.
- Advice: Try values between `50` and `200`.

</details>

Keep in mind that the results of applying `t-SNE` on the same features are different every time.

- Piedmont wines, 3 runs, 3 different scatter plots, same data used:

![w08_tsne_ex2.png](assets/w08_tsne_ex2.png "w08_tsne_ex2.png")

## Principal Component Analysis (PCA)

PCA is a technique for reducing dimensionality by finding new axes (principal components) that capture the maximum variance in the data.

- Benefits:
  - **Compression**: Represent data with fewer dimensions while preserving most variance.
  - **Efficiency**: Faster computation and storage for large datasets.
  - **Noise Reduction**: Removes less-informative features.

- How PCA Works:
  1. **Center the Data**: Subtract the mean of each feature.
  2. **Compute Principal Components**: Find directions (orthogonal axes) that maximize variance using eigen decomposition or SVD.
  3. **Project Data**: Transform original data onto these new axes.
  4. **Optional**: Reduce dimensions by keeping only the top `k` components.

> **Note:** PCA does **not scale features by default**. If features have different scales, use `StandardScaler` before PCA.

- [In scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#pca):
  - `fit()`: Learns principal components from data.
  - `transform()`: Projects data onto these components.
  - `whiten=True`: Scales components to unit variance (useful for algorithms assuming isotropy, e.g., SVM with RBF kernel, K-Means).

![w08_pca_ex1.png](assets/w08_pca_ex1.png "w08_pca_ex1.png")

> **Note:** After rotation into the principal component basis, the covariance matrix becomes diagonal, meaning the new features (components) are uncorrelated.

<details>

<summary>So, what are principal components?</summary>

The directions in which the samples vary the most.

</details>

The resulting table has the PCs ordered in descending order of their amount of variance. It is also known as the `eigenvector matrix`.

![w08_pca_picture.png](assets/w08_pca_picture.png "w08_pca_picture.png")

- Available as `components_` attribute of PCA object.
- Each row defines displacement from mean.

```python
print(model.components_)
```

```console
[[ 0.64116665  0.76740167]
 [-0.76740167  0.64116665]]
```

Although, in `sklearn` the default behavior is not to standardize the data, the textbook PCA algorithm includes scaling. Thus, it is always recommended to scale the data before passing it to PCA:

![w08_pca_in_action.gif](assets/w08_pca_in_action.gif "w08_pca_in_action.gif")

Three scatter plots of the same point cloud are shown. Each scatter plot shows a different set of axes (in red). In which of the plots could the axes represent the principal components of the point cloud?

![w08_pca_checkpoint.png](assets/w08_pca_checkpoint.png "w08_pca_checkpoint.png")

<details>

<summary>Reveal answer</summary>

Answer: Both plot $1$ and plot $3$.

</details>

Let's consider $2$ features: latitude and longitude at points along a flight path:

```console
latitude  longitude
  50.529     41.513
  50.360     41.672
  50.196     41.835
...
```

<details>

<summary>What is the dimension of the dataset?</summary>

- Dataset *appears* to be `2`-dimensional.
- But can approximated using only `1` feature: `displacement` along flight path:

![w08_intrinsic_dim.png](assets/w08_intrinsic_dim.png "w08_intrinsic_dim.png")

- So, the dataset is intrinsically `1`-dimensional.

</details>

<details>

<summary>How can we define the term intrinsic dimension?</summary>

- Intrinsic dimension = number of features needed to approximate the dataset.
- Essential idea behind dimensionality reduction.
- Answers the question: *What is the most compact representation of the samples?*.
- Can be detected with PCA.

</details>

Let's filter the iris dataset and leave only the `versicolor` samples.

![w08_intrinsic_dim_iris.png](assets/w08_intrinsic_dim_iris.png "w08_intrinsic_dim_iris.png")

<details>

<summary>What is the intrinsic dimension of the dataset?</summary>

If we plot the $3$ features that measure them, we see that the samples lie close to a flat $2$-dimensional plane. So, they can be approximated using $2$ features.

</details>

- Scatter plots only work if samples have $2$ or $3$ features.
- PCA identifies intrinsic dimension when samples have any number of features.
- Intrinsic dimension = number of PCA features with significant variance.

![w08_intrinsic_dim_pca1.png](assets/w08_intrinsic_dim_pca1.png "w08_intrinsic_dim_pca1.png")

- PCA features are ordered by decreasing variance.
- In our example, these are the first $2$ PCA features.
- Here's a barplot with their values. You can obtain them by using the attribute `explained_variance_` of a fitted `PCA` object.

![w08_intrinsic_dim_pca2.png](assets/w08_intrinsic_dim_pca2.png "w08_intrinsic_dim_pca2.png")

- Intrinsic dimension is an idealization.
- There is not always one correct answer.
- Piedmont wines: could argue for `2`, or `3`, or more.

![w08_intrinsic_dim_pca3.png](assets/w08_intrinsic_dim_pca3.png "w08_intrinsic_dim_pca3.png")

<details>

<summary>If you had to summarize what you learned about PCA what would they be?</summary>

- Used to represent the same data, using less features.
- Orders the features in decreasing order of variance.
- Assumes the low variance features are `noise`.
- Assumes the high variance features are informative.

![w08_pca_summary.png](assets/w08_pca_summary.png "w08_pca_summary.png")

</details>

## `TruncatedSVD` and `TF-IDF`

- Alternative implementation of PCA that does not center the data.
- Typically used for word-frequency arrays:
  - Rows represent documents, columns represent words.
  - Entries measure presence of each word in each document:
    - Measurement can be count, tf-idf, etc.

<details>

<summary>What would be the most likely value to see in a word frequency array?</summary>

Only some of the words from the vocabulary appear in any one document, so most of the entries are `0`.

![w08_wfa.png](assets/w08_wfa.png "w08_wfa.png")

</details>

<details>

<summary>How are such arrays called?</summary>

- Arrays in which most entries are `0`, are called `sparse` arrays.
- They can be represented using a special type of an array, called [`csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) (Compressed Sparse Row matrix).
  - ["Remembers"](https://en.wikipedia.org/wiki/Sparse_matrix) only the non-zero entries.

![w08_tf_idf.png](assets/w08_tf_idf.png "w08_tf_idf.png")

</details>

- scikit-learn's `PCA` is not efficient for `csr_matrix`, but [`TruncatedSVD`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) is!
- Use `TruncatedSVD` when working with sparse matrices. Its interface is the same as the one `PCA` has.

<details>

<summary>What does tf-idf measure?</summary>

It is used in text mining and information retrieval to measure how important a word is in a document relative to a collection of documents (corpus).

</details>

It combines two metrics: term frequency and inverse document frequency.

<details>

<summary>What is term frequency?</summary>

Measures how often a term appears in a document.

</details>

<details>

<summary>What is formula for term frequency?</summary>

$$\text{TF}(t, d) = \frac{\text{Number of times term t appears in document d}​}{\text{Total number of terms in document d}}$$

</details>

<details>

<summary>What does high TF mean?</summary>

The term is frequent in that document.

</details>

<details>

<summary>What is inverse document frequency?</summary>

Measures how unique or rare a term is across all documents.

</details>

<details>

<summary>What is formula for inverse document frequency?</summary>

$$\text{IDF}(t) = \ln \left[ \frac{N​}{1 + \text{DF}(t)} \right]$$

Where:

- $N$ = total number of documents in the corpus.
- $\text{DF}(t)$ = number of documents containing term $t$.
- Adding $1$ in the denominator avoids division by zero.

</details>

<details>

<summary>What would be the inverse document frequency of common words like like "the", "and", "we", "I", etc?</summary>

They would have very low IDF since $\ln(1) = 0$.

</details>

<details>

<summary>Do you know how such common words are referred to?</summary>

These words are known as `stopwords` because they do have much added value to understanding a document (as all documents have them).

</details>

<details>

<summary>Which words would have a high inverse document frequency?</summary>

Rare words.

</details>

When we combine the two metrics we get:

$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)$$

## Non-negative matrix factorization (NMF)

- Dimension reduction technique.
- NMF models are interpretable (unlike PCA), i.e. easy to explain.
- One requirement: all sample features must be non-negative.
- It expresses documents as combinations of topics (or `themes`):

![w08_nmf_1.png](assets/w08_nmf_1.png "w08_nmf_1.png")

- and images as combination of patterns:

![w08_nmf_2.png](assets/w08_nmf_2.png "w08_nmf_2.png")

- In scikit-learn it's implemented as [`NMF`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html) and follows the standard `fit()` and `transform()` pattern.
- Works with normal (non-sparse) NumPy arrays and with `csr_matrix`.

### NMF features

- NMF features are also non-negative.
- Can be used to reconstruct the samples by ***matrix multiplying*** them with the identified components.
- This is the "**M**atrix **F**actorization" part in `NMF`.

```python
print(samples[i])
```

```console
[  0.12  0.18  0.32  0.14]
```

```python
print(nmf_features[i])
```

```console
[  0.15  0.12]
```

![w08_nmf_4.png](assets/w08_nmf_4.png "w08_nmf_4.png")

Which of the following 2-dimensional arrays are examples of non-negative data?

1. A tf-idf word-frequency array.
2. An array daily stock market price movements (up and down), where each row represents a company.
3. An array where rows are customers, columns are products and entries are 0 or 1, indicating whether a customer has purchased a product.

<details>

<summary>Reveal answer</summary>

Answer: `1` and `3`

Stock prices can go down as well as up, so an array of daily stock market price movements is not an example of non-negative data.

</details>

The components of an NMF model are given. If the NMF feature values of a sample are `[2, 1]`, then which of the following is most likely to represent the original sample?

```python
[[1, 0.5, 0], [0.2, 0.1, 2.1]]
```

1. `[2.2, 1.1, 2.1]`.
2. `[0.5, 1.6, 3.1]`.
3. `[-4.0, 1.0, -2.0]`.

<details>

<summary>Reveal answer</summary>

Answer: `1`.

We have matrix multiply the features with the components:

```python
nmf_features = np.array([2, 1])
components = np.array([[1, 0.5, 0], [0.2, 0.1, 2.1]])
np.dot(nmf_features, components)
```

</details>

### NMF components are topics

Example:

- Word-frequency array of articles (tf-idf).
- 20,000 scientific articles (rows).
- 800 words (columns).

![w08_nmf_5.png](assets/w08_nmf_5.png "w08_nmf_5.png")

If we apply NMF with $10$ components to this data, we would get a resulting matrix with shape $(10, 800)$.

Choosing a component and looking at which words have the highest values, we see that they fit a theme: the words are `species`, `plant`, `plants`, `genetic`, `evolution` and `life`.

![w08_nmf_6.png](assets/w08_nmf_6.png "w08_nmf_6.png")

The same happens if any other component is considered.

![w08_nmf_7.png](assets/w08_nmf_7.png "w08_nmf_7.png")

So, when applied to text-based documents:

- `NMF` components represent topics;
- `NMF` features combine topics into documents.

For images, NMF components are parts of images:

![w08_nmf_8.png](assets/w08_nmf_8.png "w08_nmf_8.png")

<details>

<summary>How can we represent a collection of images as a non-negative array?</summary>

Answer: Using their pixels.

### Grayscale images

- Grayscale image = no colors, only shades of gray.
- They can be used to measure pixel brightness by having a value between `0` and `1` (`0` is black).
- Here's an 8x8 grayscale image of the moon, written as an array:

![w08_nmf_9.png](assets/w08_nmf_9.png "w08_nmf_9.png")

### Encoding a collection of images

- Encode each image as 1D flat array.
- Encode the collection as 2D array.
- Each row corresponds to an image.
- Each column corresponds to a pixel.
- We can then apply NMF.

![w08_nmf_10.png](assets/w08_nmf_10.png "w08_nmf_10.png")
</details>

### Building recommender systems using NMF

Image you're a data scientist at a large online newspaper.

You've been tasked to recommend articles similar to the article being currently read by a customer.

<details>

<summary>How would you go about doing this?</summary>

Similar articles should have similar topics!

1. Convert the articles into a word-frequency array.
2. We can extract the different topics by applying NMF to the array.
3. The extracted topics themselves are just vectors.
4. We can compute how close those vectors are to one another, by using the consine similarity.
5. Whichever vectors are the closest to the one obtained by the current article, would represent the articles we should return to the user.

Recap of how cosine similarity works:

![w08_consine_similarity.png](assets/w08_consine_similarity.png "w08_consine_similarity.png")

- Uses the angle between the lines.
- Higher values means more similar.
- Maximum value is `1`, when angle is `0` degrees.

Here's how this can be done via `scikit-learn`:

```python
from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)
# if the current document has index 23
current_article = norm_features[23]
similarities = norm_features.dot(current_article)
print(similarities)
```

```console
[ 0.7150569  0.26349967 ..., 0.20323616  0.05047817]
```

</details>

# Week 09 - Natural Language Processing

## The added value of NLP

<details>

<summary>What is natural language processing?</summary>

Field of study focused on making sense of language using statistics and computers.

</details>

<details>

<summary>Why is it important?</summary>

It lets us model characteristics holding text by turning them into numbers.

</details>

<details>

<summary>What applications of natural language processing techniques have you heard of?</summary>

- Chatbots.
- Translation.
- Sentiment analysis.
- Named-entity recognition.
- Tokenization.
- etc, etc.

</details>

## Regular expressions

<details>

<summary>What are regular expressions?</summary>

Strings with a special syntax that allow us to match patterns in other strings.

</details>

<details>

<summary>Why do we use them?</summary>

To parse the important pieces of information in large text corpora that may be **unstructured or noisy**. They help us identify patterns like dates, phone numbers, email addresses, or specific keywords without manually scanning the whole text.

</details>

<details>

<summary>Do you know what a "corpus" (plural, "corpora") is?</summary>

A set of raw texts (**not** preprocessed!) to perform NLP tasks on.

</details>

<details>

<summary>What application of regular expressions have you heard of?</summary>

- Find all web links in a document.
- Parse email addresses.
- Remove/replace unwanted characters.
- Validate user credentials.
- Part of speech tagging.

</details>

<details>

<summary>Do you know the name of a very popular website for playing around with regular expressions?</summary>

[regex101](https://regex101.com/)!

</details>

<details>

<summary>Which package in Python has functions for working with regular expression?</summary>

Regular expressions can be used in Python via the [`re`](https://docs.python.org/3/library/re.html) library.

</details>

Open the package and let's answer some questions.

<details>

<summary>What does the function "match" do?</summary>

- `re.match`: matches a pattern with a string:
  - first argument is the pattern;
  - second argument is the string in which to match the pattern.
  - Returns a [`Match`](https://docs.python.org/3/library/re.html#re.Match) object and `None` if the string does not match the pattern.

</details>

What would be returned in the following case:

```python
import re
re.match('abc', 'abcdef')
```

<details>

<summary>Reveal answer</summary>

```console
<re.Match object; span=(0, 3), match='abc'>
```

</details>

What about this:

```python
re.match(r'\w+', 'hi there!') # notice the r-string used here!
```

<details>

<summary>Reveal answer</summary>

```console
<re.Match object; span=(0, 2), match='hi'>
```

</details>

<details>

<summary>How can we get the actual text from a Match object?</summary>

We can use `.group()`:

```python
re.match(r'\w+', 'hi there!').group()
```

```console
'hi'
```

</details>

<details>

<summary>What is the meaning of strings with prefix "r"?</summary>

These are the so-called a `raw strings`.

- Without the prefix, backslashes ($\$) are treated as escape characters (e.g., `'\n'` becomes a newline).
- With prefix `'r'` backslashes are not interpreted as escapes, so the string is taken literally.

```python
# Without raw string
pattern = "\\d+"   # Matches digits

# With raw string
pattern = r"\d+"   # Matches digits (cleaner)
```

</details>

Fill in the below patterns:

| Pattern | Type            | Description | Example Text        | Matches in Example                     |
| ------- | --------------- | ----------- | ------------------- | -------------------------------------- |
| ???     | Character class | ???         | `"Magic is real"`   | `Magic`, `is`, `real`                  |
| ???     | Character class | ???         | `"Room 9 is ready"` | `9`                                    |
| ???     | Character class | ???         | `"Hello World"`     | the space between the words            |
| ???     | Quantifier      | ???         | `"username74"`      | `username74`                           |
| ???     | Quantifier      | ???         | `"AAACC"`           | `['CC']` (for `r'C+'`)                 |
| ???     | Quantifier      | ???         | `"AAACC"`           | `['', '', '', 'CC', '']` (for `r'C*'`) |
| ???     | Character class | ???         | `"no_spaces"`       | `n`, `o`, `_`, etc.                    |
| ???     | Character set   | ???         | `"abcdefg"`         | `a`, `b`, `c`, ...                     |

<details>

<summary>Reveal answer</summary>

| Pattern | Type            | Description                                                   | Example Text        | Matches in Example                     |
| ------- | --------------- | ------------------------------------------------------------- | ------------------- | -------------------------------------- |
| `\w+`   | Character class | One or more word characters (letters, digits, and underscore) | `"Magic is real"`   | `Magic`, `is`, `real`                  |
| `\d`    | Character class | A single digit                                                | `"Room 9 is ready"` | `9`                                    |
| `\s`    | Character class | A whitespace character (space, tab, newline)                  | `"Hello World"`     | the space between the words            |
| `.*`    | Quantifier      | Zero or more of any character (except newline)                | `"username74"`      | `username74`                           |
| `+`     | Quantifier      | One or more repetitions                                       | `"AAACC"`           | `['CC']` (for `r'C+'`)                 |
| `*`     | Quantifier      | Zero or more repetitions                                      | `"AAACC"`           | `['', '', '', 'CC', '']` (for `r'C*'`) |
| `\S`    | Character class | A non-whitespace character                                    | `"no_spaces"`       | `n`, `o`, `_`, etc.                    |
| `[a-z]` | Character set   | Any lowercase letter                                          | `"abcdefg"`         | `a`, `b`, `c`, ...                     |

</details>

Fill in the below functions:

| Function | Purpose | Example Code                            | Output Example              |
| -------- | ------- | --------------------------------------- | --------------------------- |
| ???      | ???     | `re.???(r'\s+', 'Kick on spaces.')`     | `['Kick', 'on', 'spaces.']` |
| ???      | ???     | `re.???(r'\d+', 'Order 123, item 456')` | `['123', '456']`            |
| ???      | ???     | `re.???(r'\d+', 'Order 123')`           | `<Match object: '123'>`     |
| ???      | ???     | `re.???(r'Order', 'Order 123')`         | `<Match object: 'Order'>`   |

<details>

<summary>Reveal answer</summary>

| Function  | Purpose                                 | Example Code                                | Output Example              |
| --------- | --------------------------------------- | ------------------------------------------- | --------------------------- |
| `split`   | Split a string based on a regex pattern | `re.split(r'\s+', 'Kick on spaces.')`       | `['Kick', 'on', 'spaces.']` |
| `findall` | Find all occurrences of a pattern       | `re.findall(r'\d+', 'Order 123, item 456')` | `['123', '456']`            |
| `search`  | Search for first match in a string      | `re.search(r'\d+', 'Order 123')`            | `<Match object: '123'>`     |
| `match`   | Match pattern at the start of a string  | `re.match(r'Order', 'Order 123')`           | `<Match object: 'Order'>`   |

</details>

Which of the following Regex patterns results in the following text?

```python
>>> my_string = "Let's write RegEx!"
>>> re.findall(PATTERN, my_string)
['Let', 's', 'write', 'RegEx']
```

```text
A. r"\s+"
B. r"\w+"
C. r"[a-z]"
D. r"\w"
```

<details>

<summary>Reveal answer</summary>

Answer: B.

</details>

<details>

<summary>What is the difference between `re.search()` and `re.match()`?</summary>

The main difference is what index they start searching from:

- `search` scans through the whole string (searching starts from every index);
- `match` scans only the beginning of the string (searching starts only from the first index).

```python
re.match(r'cd', 'abcde')
```

```console
# Nothing is returned, i.e. empty string
```

```python
re.search(r'cd', 'abcde')
```

```console
<re.Match object; span=(2, 4), match='cd'>
```

> **Note:** Both return the first matching substring.

</details>

<details>

<summary>How can we place multiple patterns in a single expression? What if we wanted to match words or numbers?</summary>

- OR is represented using `|`.
- You can define a group using `()`.
- You can define explicit character range using `[]`.

```python
import re
re.findall(r'(\d+|\w+)', 'He has 11 cats.')
```

```console
['He', 'has', '11', 'cats']
```

</details>

Fill in the below patterns:

| pattern | matches                                           | example            |
| ------- | ------------------------------------------------- | ------------------ |
| ???     | upper and lowercase English alphabet              | 'ABCDEFghijk'      |
| ???     | numbers from 0 to 9                               | 9                  |
| ???     | upper and lowercase English alphabet, `-` and `.` | `'My-Website.com'` |
| ???     | `a`, `-` and `z`                                  | 'a-z'              |
| ???     | spaces or a comma                                 | `', '`             |

<details>

<summary>Reveal answer</summary>

| pattern        | matches                                           | example            |
| -------------- | ------------------------------------------------- | ------------------ |
| `[a-zA-Z]+`    | upper and lowercase English alphabet              | `'ABCDEFghijk'`    |
| `[0-9]`        | numbers from 0 to 9                               | `9`                |
| `[a-zA-Z\-.]+` | upper and lowercase English alphabet, `-` and `.` | `'My-Website.com'` |
| `[a\-z]`       | `a`, `-` and `z`                                  | `'a-z'`            |
| `[\s,]+`       | spaces or a comma                                 | `', '`             |

</details>

## Tokenization

Tokenization is often the first step in preparing a text for NLP.

<details>

<summary>What is tokenization?</summary>

Turning a string or document into smaller chunks. Those chunks are called **tokens**.

</details>

<details>

<summary>Why should we tokenize text?</summary>

- Breaking our words or sentences.
- Separating punctuation.
- Separating all hashtags in a tweet.
- Easier to map part of speech.
- Removing common words.
- Removing unwanted tokens.

</details>

There are many different types of tokenization and rules for doing it. We can create our own rules using regular expressions.

<details>

<summary>What Python library is used for basic natural language processing?</summary>

[nltk](https://www.nltk.org/): Natural language toolkit.

```python
import nltk
from nltk import tokenize
nltk.download('punkt_tab') # This is required only once. 
tokenize.word_tokenize('Hi there!')
```

```console
['Hi', 'there', '!']
```

Other tokenizers:

- [`sent_tokenize`](https://www.nltk.org/api/nltk.tokenize.sent_tokenize.html): tokenize a document into sentences;
- [RegexpTokenizer](https://www.nltk.org/api/nltk.tokenize.regexp.html): tokenize a string or document based on a regular expression pattern;
- [TweetTokenizer](https://www.nltk.org/api/nltk.tokenize.casual.html#nltk.tokenize.casual.TweetTokenizer): special class just for tweet tokenization, allowing you to separate hashtags, mentions and lots of exclamation points.

</details>

Here is a string:

```python
my_string = "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"
```

Our goal would be to get the following tokenization:

```console
['SOLDIER', '#1', 'Found', 'them', '?', 'In', 'Mercea', '?', 'The', 'coconut', 's', 'tropical', '!']
```

Which of the below patterns is the best tokenizer?

A. `r"(\w+|\?|!)"`

B. `r"(\w+|#\d|\?|!)"`

C. `r"(#\d\w+\?!)"`

D. `r"\s+"`

<details>

<summary>Reveal answer</summary>

Answer: B.

You can try it out:

```python
import re
my_string = "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"
re.findall(r"(\w+|#\d|\?|!)", my_string)
```

</details>

In deep learning tasks tokenization is usually followed by transforming each token into a word-vector.

<details>

<summary>Do you know what word vectors are?</summary>

It is a word that has been ***embedded*** in an `n` dimensional **semantic** space, so `word vectors = n-dimensional vector`. The words around it have similar meaning to it. Words far from it have a different meaning.

![w09_word_vectors.png](assets/w09_word_vectors.png "w09_word_vectors.png")

- With quality word vectors (creating using neural networks) we can make the following conclusions:
  - The vector operation `king minus queen` is approximately equal to `man minus woman`.
  - Spain is to Madrid as Italy is to Rome.

</details>

## Bag-of-words

<details>

<summary>What is bag-of-words?</summary>

Creating a matrix in which:

- the rows are documents;
- the columns are words;
- the elements inside are counts.

</details>

<details>

<summary>What are the two main steps of the bag-of-words approach for a single document?</summary>

1. Tokenize input.
2. Count up all the tokens.

</details>

<details>

<summary>What is an "important token" in the context of bag-of-words?</summary>

The more frequent a token is, the more important it might be.

Thus, this approach can be a good (not the greatest) way to determine the significant words in a document.

</details>

<details>

<summary>In what case would this approach for word importance be problematic?</summary>

When we have the so-called stopwords: `"the", "and", "I", "but", etc`. Similar words may not be that important and may introduce noise.

</details>

Here's some text:

```python
'The cat is in the box. The cat likes the box. The box is over the cat.'
```

<details>

<summary>What would bag-of-words produce if we were to strip the punctuation?</summary>

- "The": 3
- "box": 3
- "cat": 3
- "the": 3
- "is": 2
- "in": 1
- "likes": 1
- "over": 1

</details>

<details>

<summary>How can we do this in Python?</summary>

The Python [Counter](https://docs.python.org/3/library/collections.html#collections.Counter) can be really helpful here:

```python
import collections
from nltk import tokenize
counter = collections.Counter(tokenize.word_tokenize('The cat is in the box. The cat likes the box. The box is over the cat.'))
counter
```

```console
Counter({'The': 3, 'cat': 3, 'the': 3, 'box': 3, '.': 3, 'is': 2, 'in': 1, 'likes': 1, 'over': 1})
```

It also has a pretty useful method - `most_common`:

```python
counter.most_common(2)
```

```console
[('The', 3), ('cat', 3)]
```

</details>

- `sklearn` has its own implementation in the class [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html):
  - Turns a text into bag-of-words vectors.
  - Can also remove stopwords via its `stop_words='english'` parameter setting.
    - Same applies to removing punctuation.
  - Returns `scipy.sparse.csr_matrix`.

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
vectorizer.get_feature_names_out()
```

```console
[[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]
 [0 1 1 1 0 0 1 0 1]]
array(['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third',
       'this'], dtype=object)
```

We can also create bigrams:

```python
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X2 = vectorizer2.fit_transform(corpus)
print(X2.toarray())
vectorizer2.get_feature_names_out()
```

```console
[[0 0 1 1 0 0 1 0 0 0 0 1 0]
 [0 1 0 1 0 1 0 1 0 0 1 0 0]
 [1 0 0 1 0 0 0 0 1 1 0 1 0]
 [0 0 1 0 1 0 1 0 0 0 0 0 1]]
array(['and this', 'document is', 'first document', 'is the', 'is this',
       'second document', 'the first', 'the second', 'the third',
       'third one', 'this document', 'this is', 'this the'], dtype=object)
```

Which of the below options, with basic `nltk` tokenization, maps the bag-of-words for the following text?

`"The cat is in the box. The cat box."`

```text
A. ('the', 3), ('box.', 2), ('cat', 2), ('is', 1)
B. ('The', 3), ('box', 2), ('cat', 2), ('is', 1), ('in', 1), ('.', 1)
C. ('the', 3), ('cat box', 1), ('cat', 1), ('box', 1), ('is', 1), ('in', 1)
D. ('The', 2), ('box', 2), ('.', 2), ('cat', 2), ('is', 1), ('in', 1), ('the', 1)
```

<details>

<summary>Reveal answer</summary>

Answer: D.

</details>

Let's now contrast this with `tf-idf`.

<details>

<summary>What does "tf" stand for?</summary>

Term frequency.

</details>

<details>

<summary>What does "idf" stand for?</summary>

Inverse document frequency.

</details>

<details>

<summary>Explain "tf-idf" - what do the different parts mean?</summary>

- `tf`: number of times a token is present in a document;
- `idf`: number of documents / number of documents that contain the token.

</details>

<details>

<summary>What is the use-case for "tf-idf"?</summary>

Allows us to determine the most important words in a document by weighing them. Notice: important `!=` common. In other words, it keeps document specific frequent words weighted high.

</details>

<details>

<summary>How does "tf-idf" treat stopwords?</summary>

It assigns lower weights to them as they are present in almost every document. This ensures most common words don't show up as key words.

</details>

<details>

<summary>What is the formula for "tf-idf"?</summary>

$$w_{i,j} = tf_{i,j} * ln \left[ \frac{N}{1 + df_i} \right]$$

, where:

- $tf_{i,j} =$ number of occurrences of token $i$ in document $j$.
- $N =$ total number of documents.
- $df_{i,j} =$ number of documents that contain token $i$.
- $w_{i,j} =$ tf-df weight for token $i$ in document $j$.

</details>

You want to calculate the tf-idf weight for the word `computer`, which appears five times in a document containing a hundred words. Given a corpus containing two hundred documents, with twenty documents mentioning the word `computer`, which of the below options is correct?

```text
A. (5 / 100) * ln(200 / 20)
B. (5 * 100) / ln(200 * 20)
C. (20 / 5) * ln(200 / 20)
D. (200 * 5) * ln(400 / 5)
```

<details>

<summary>Reveal answer</summary>

Answer: A.

</details>

## Text preprocessing

<details>

<summary>Why preprocess at all?</summary>

- Helps make for better input data.
- Increases quality of data.
- Removes noise - commonly used words `"the"`, `"and"` and similar may not be that important.

</details>

<details>

<summary>Have you heard of any examples of preprocessing?</summary>

- Tokenization to create a bag-of-words.
- Lowercasing.
- Removing `stopwords` - `"the"`, `"and"`, `"but"`, `"a"`, `"on"`, `"at"`, etc.

```python
import nltk
nltk.download('stopwords')
nltk.corpus.stopwords.words('english')
```

```console
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
```

- Removing punctuation.
- Removing other unwanted tokens.

</details>

<details>

<summary>What are stemming and lemmatization? What are the problems they solve?</summary>

![w09_stemming_vs_lemmatization.png](assets/w09_stemming_vs_lemmatization.png "w09_stemming_vs_lemmatization.png")

`nltk.stem` provides implementations for:

- Lemmatization: [WordNetLemmatizer](https://www.nltk.org/api/nltk.stem.WordNetLemmatizer.html?highlight=wordnet#nltk-stem-wordnetlemmatizer);
- Stemming: [PorterStemmer](https://www.nltk.org/api/nltk.stem.PorterStemmer.html).

</details>

<details>

<summary>How do we choose which processing techniques to apply?</summary>

That is what the data audit is for!

When we have the task in mind (classification, regression, something else) we can intuit what text features might be useful and how to obtain them.

As always the best approach is discovered via multiple experiments of using a single or a combination of techniques and choosing the one(s) leading to the highest metrics.

</details>

<details>

<summary>If we have to classify whether a review of a hotel is positive or negative, would the removal of stopwords be helpful?</summary>

Most definitely! We're interested in discovering what combination of words lead to positive sentiment and what to negative.

</details>

<details>

<summary>When would the removal of stopwords be harmful?</summary>

For example, when we want to discover authorship styles. The task about [Author profiling](https://en.wikipedia.org/wiki/Author_profiling) is focused precisely on finding out what tokens an author uses and using them to attribute texts to them.

</details>

Which of the following are all useful text preprocessing steps?

```text
A. Stems, spelling corrections, lowercase.
B. Lemmatization, lowercasing, removing unwanted tokens.
C. Removing stop words, leaving in capital words.
D. Strip stop words, word endings and digits.
```

<details>

<summary>Reveal answer</summary>

Answer: B.

</details>

What are word vectors and how do they help with NLP?

```text
A. They are similar to bags of words, just with numbers. You use them to count how many tokens there are.
B. Word vectors are sparse arrays representing bigrams in the corpora. You can use them to compare two sets of words to one another.
C. Word vectors are multi-dimensional mathematical representations of words created using deep learning methods. They give us insight into relationships between words in a corpus.
D. Word vectors don't actually help NLP and are just hype.
```

<details>

<summary>Reveal answer</summary>

Answer: C.

</details>

## [Gensim](https://radimrehurek.com/gensim/auto_examples/index.html#documentation)

- Popular open-source NLP library.
- A more powerful alternative to `nltk`.
- Can be used to perform complex NLP tasks:
  - Building document vectors.
  - Building word vectors.
  - Topic identification.
  - Document comparison.
- `gensim` objects can be easily saved, updated and reused.

Gensim allows you to build corpora and dictionaries using classes and functions.

Open the documentation and answer the following questions.

<details>

<summary>How can we create a mapping with a numeric identifier for each token?</summary>

We can use the class `Dictionary` class:

```python
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize

my_documents = ['The movie was about a spaceship and about aliens.','I really liked the movie!','Awesome action scenes, but boring characters.','The movie was awful! I hate alien films.','Space is cool! I liked the movie.','More space films, please!',]

tokenized_docs = [word_tokenize(doc.lower()) for doc in my_documents]
dictionary = Dictionary(tokenized_docs)
dictionary.token2id
```

```console
{'.': 0, 'a': 1, 'about': 2, 'aliens': 3, 'and': 4, 'movie': 5, 'spaceship': 6, 'the': 7, 'was': 8, '!': 9, 'i': 10, 'liked': 11, 'really': 12, ',': 13, 'action': 14, 'awesome': 15, 'boring': 16, 'but': 17, 'characters': 18, 'scenes': 19, 'alien': 20, 'awful': 21, 'films': 22, 'hate': 23, 'cool': 24, 'is': 25, 'space': 26, 'more': 27, 'please': 28}
```

</details>

We now can represent whole documents using:

- A list of their token ids.
- How often those tokens appear in each document.

<details>

<summary>Which method converts a document into the bag-of-words format?</summary>

The method `doc2bow` converts a document into the format `(token_id, token_count)`, sorted by `token_id`:

```python
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
corpus
```

```console
[[(0, 1), (1, 1), (2, 2), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)], [(5, 1), (7, 1), (9, 1), (10, 1), (11, 1), (12, 1)], [(0, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1)], [(0, 1), (5, 1), (7, 1), (8, 1), (9, 1), (10, 1), (20, 1), (21, 1), (22, 1), (23, 1)], [(0, 1), (5, 1), (7, 1), (9, 1), (10, 1), (11, 1), (24, 1), (25, 1), (26, 1)], [(9, 1), (13, 1), (22, 1), (26, 1), (27, 1), (28, 1)]]
```

</details>

## Named Entity Recognition

<details>

<summary>What is named entity recognition (NER)?</summary>

- NLP task to identify important named entities in a text:
  - people;
  - places;
  - organizations;
  - dates;
  - states;
  - works of art;
  - etc., etc.
- Can be used alongside topic identification.
- Main use-case for NER is answering questions related to a text:
  - Who?
  - What?
  - When?
  - Where?
- Give an example with the `payee model`.

</details>

Here's an example:

![w09_ner.png](assets/w09_ner.png "w09_ner.png")

### Using `nltk`

```python
import nltk
nltk.download('averaged_perceptron_tagger_eng')
sentence = '''In New York, I like to ride the Metro to
              visit MOMA and some restaurants rated
              well by Ruth Reichl.'''
tokenized_sent = nltk.word_tokenize(sentence)
tagged_sent = nltk.pos_tag(tokenized_sent)
tagged_sent[:3]
```

```console
[('In', 'IN'), ('New', 'NNP'), ('York', 'NNP')]
```

To get information about the various POS (part-of-speech) tags, use the following:

```python
import nltk
nltk.download('tagsets_json')
nltk.help.upenn_tagset()
```

```console
$: dollar
    $ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$
'': closing quotation mark
    ' ''
(: opening parenthesis
    ( [ {
): closing parenthesis
    ) ] }
,: comma
    ,
--: dash
    --
.: sentence terminator
    . ! ?
:: colon or ellipsis
    : ; ...
CC: conjunction, coordinating
    & 'n and both but either et for less minus neither nor or plus so
    therefore times v. versus vs. whether yet
CD: numeral, cardinal
    mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-
    seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025
    fifteen 271,124 dozen quintillion DM2,000 ...
DT: determiner
    all an another any both del each either every half la many much nary
    neither no some such that the them these this those
EX: existential there
    there
FW: foreign word
    gemeinschaft hund ich jeux habeas Haementeria Herr K'ang-si vous
    lutihaw alai je jour objets salutaris fille quibusdam pas trop Monte
    terram fiche oui corporis ...
...
```

```python
import nltk
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
sentence = '''In New York, I like to ride the Metro to
              visit MOMA and some restaurants rated
              well by Ruth Reichl.'''
tokenized_sent = nltk.word_tokenize(sentence)
tagged_sent = nltk.pos_tag(tokenized_sent)
print(nltk.ne_chunk(tagged_sent))
```

```console
(S
  In/IN
  (GPE New/NNP York/NNP)
  ,/,
  I/PRP
  like/VBP
  to/TO
  ride/VB
  the/DT
  (ORGANIZATION Metro/NNP)
  to/TO
  visit/VB
  (ORGANIZATION MOMA/NNP)
  and/CC
  some/DT
  restaurants/NNS
  rated/VBN
  well/RB
  by/IN
  (PERSON Ruth/NNP Reichl/NNP)
  ./.)
```

### What is [SpaCy](https://spacy.io/)?

- NLP library, similar to `gensim`, with different implementations.
- Open-source, with extra libraries and tools:
  - [Displacy](https://demos.explosion.ai/displacy): visualization tool for viewing parse trees which uses Node-js to create interactive text.
  - [Displacy Entity Recognition Visualizer](https://demos.explosion.ai/displacy-ent/).
- Easy creation of multi-step pipelines.
- Has different entity types compared to `nltk`.
- Great informal language corpora: easily find entities in Tweets and chat messages.

```python
import spacy
nlp = spacy.load("en_core_web_sm")
nlp.get_pipe("ner").labels
```

```console
('CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART')
```

```python
import spacy
nlp_pipeline = spacy.load('en_core_web_sm')
nlp_pipeline.get_pipe('ner')
```

```console
<spacy.pipeline.ner.EntityRecognizer object at 0x796d209882e0>
```

```python
doc = nlp_pipeline('''Berlin is the capital of Germany;
                           and the residence of Chancellor Angela Merkel.''')
doc.ents
```

```console
(Berlin, Germany, Angela Merkel)
```

```python
doc.ents[0], doc.ents[0].label_
```

```console
(Berlin, 'GPE')
```

## The Naive Bayes Classifier

- Based on probability.
- Simple and effective, but naive because it assumes all features are completely independent/uncorrelated.
- Given a particular piece of data, how likely is a particular outcome?
  - If a film's plot has a spaceship, how likely is it to be a sci-fi?
  - Given a spaceship **and** an alien, how likely **now** is it sci-fi?

Imagine we receive:

- **normal emails** from friends and family;
- and we also receive some **spam email**: unwanted emails that are usually scams or unsolicited advertisements.

We don't want the spam messages so we try to create an algorithm that filters them out.

1. We tokenize all of the emails and get a list of tokens.
2. For each of our classes $C_k$ (here $k = 2$ since we have normal and spam emails):
   1. We calculate its **prior probability**:
      $$P(C_k) = \frac{|C_k|}{\sum_i |C_i|}$$
   2. And the **likelihood** of seeing each word $w_i$:
      $$P(w_i | C_k) = \frac{|\{{w_i | w \in C_k, w = w_i}\}|}{|C_k|}$$
3. For any new email, we predict the class with the highest probability:

$$\displaystyle {\hat {y}}={\underset {k\in \{1,\ldots ,K\}}{\operatorname {argmax} }}\ P(C_{k})\displaystyle \prod _{i=1}^{n}P(w_{i}\mid C_{k}).$$

Let's say we have the following probabilities:

- $P(\text{Dear} | N) = 0.47$
- $P(\text{Friend} | N) = 0.29$
- $P(\text{Lunch} | N) = 0.18$
- $P(\text{Money} | N) = 0.06$
- $P(\text{Dear} | S) = 0.29$
- $P(\text{Friend} | S) = 0.14$
- $P(\text{Lunch} | S) = 0$
- $P(\text{Money} | S) = 0.57$

And we have a total of $12$ emails, $8$ of which are non-spam.

<details>

<summary>What type of email would an email having the text "Dear Friend" be?</summary>

We calculate the likelihood the email is normal if it has those words:

$$P(N) = \frac{8}{8 + 4} = 0.67$$
$$P(N) * P(\text{Dear} | N) * P(\text{Friend} | N) = 0.67 * 0.47 * 0.29 = 0.09 \propto P(N | \text{Dear Friend})$$

And the likelihood that it'd be spam:

$$P(S) = \frac{4}{8 + 4} = 0.33$$
$$P(S) * P(\text{Dear} | N) * P(\text{Friend} | N) = 0.33 * 0.29 * 0.14 = 0.01 \propto P(S | \text{Dear Friend})$$

Because $0.09 > 0.01$, we'll predict `Normal` for this case.

</details>

Let's say we now got an email with the text `"Money Money Lunch Money Money"`.

<details>

<summary>Intuitively what type of email do you think it is?</summary>

Since initially $P(\text{Money} | S) = 0.57 > 0.06 = P(\text{Money} | N)$ it's reasonable to conclude that this is a `Spam` message.

</details>

Great! Let's do the math!

<details>

<summary>What class do we get?</summary>

We calculate the likelihood the email is normal if it has those words:

$$P(N | \text{Money Money Lunch Money Money}) = P(N) * P(\text{Lunch} | N) * P(\text{Money} | N)^4 = 0.000002$$

However, when we do the calculation for `Spam`:

$$P(S | \text{Money Money Lunch Money Money}) = P(S) * P(\text{Lunch} | S) * P(\text{Money} | S)^4 = 0$$

We get $0$ since $P(\text{Lunch} | S) = 0$!

</details>

Oops ... this is bad.

<details>

<summary>How can we get around this problem?</summary>

In practice, we usually:

- Add a small Laplace/Lidstone smoothing parameter, called $\alpha$, by default $\alpha = 1$;
- Use the logarithm for numerical stability when a lot of probabilities are multiplied.

Thus, this is the final formula:

$$\hat{y} = \underset{k}{\operatorname {argmax}} \ln P(C_k) + \sum_{i=1}^n \ln \left[ \frac{\text{count}(w_i, C_k) + \alpha}{\sum_{w \in V} \text{count}(w, C_k) + \alpha \cdot |V|} \right]$$

Where $|V|$ is the size of the vocabulary - the number of all words in the classes.

</details>

- The above is implemented in `sklearn` as [sklearn.naive_bayes.MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#multinomialnb)
- Each word from `CountVectorizer` can act as a feature since the multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as `tf-idf` also work.

<details>

<summary>So why is Naive Bayes naive?</summary>

It treat all word orders the same the score for `Dear Friend` is the same as for `Friend Dear`.

The same can be said in another way: **there is an assumption of conditional independence between every pair of features given the value of the class variable**.

Bayes’ theorem states the following relationship, given class variable $y$ and dependent feature vector $x_1$ through $x_n$,:

$$P(y \mid x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots, x_n \mid y)}
                                 {P(x_1, \dots, x_n)}$$

Using the naive conditional independence assumption that:

$$P(x_i | y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i | y),$$

for all $i$, this relationship is simplified to:

$$P(y \mid x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}
                                 {P(x_1, \dots, x_n)}$$

Since $P(x_1, \dots, x_n)$ is constant given the input, we arrive at the initial classification rule:

$$\begin{align}\begin{aligned}P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)\\\Downarrow\\\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y),\end{aligned}\end{align}$$

</details>

Note that although naive Bayes is known as a decent classifier, it is known to be a **bad estimator**, so the probability outputs from `predict_proba` are not to be taken too seriously. In other words, Naive Bayes:

- has **low variance**;
- but **high bias**.

Which of the following are possible features for a text classification problem?

```text
A. Number of words in a document.
B. Specific named entities.
C. Language.
D. All of the above.
```

<details>

<summary>Reveal answer</summary>

Answer: D.

</details>

Which of the below is the most reasonable model to use when training a new supervised model using text vector data?

```text
A. Random Forests
B. Naive Bayes
C. Linear Regression
D. Deep Learning
```

<details>

<summary>Reveal answer</summary>

Answer: B.

</details>

# Week 10 - Hello, Deep Learning. Implementing a Multilayer Perceptron

What is your experience with deep learning? Who has built a deep learning model? What was it about?

<details>

<summary>What is deep learning?</summary>

- Deep learning is a class of algorithms that solve the task of `automatic pattern recognition`.
- There are two main paradigms of programming: `imperative` and `functional`. Deep learning can be regarded as a third paradigm, different from the other two as follows: let's say you have a particular `task` you want to solve.
  - In imperative and functional programming, `you write the code directly`; you tell the machine what it has to do **explicitly** and you write exactly the code solves the task by outlining and connecting multiple steps together (i.e. **you** create an algorithm).
  - In deep learning you are **not** explicitly / directly writing the logic that would solve the task. Instead, you build a `deep learning model` that models the task you are tying to solve and **the model itself creates the algorithm** for solving your task.
- A deep learning model is a set of parameters connected in various ways. It solves tasks by finding optimal values of those parameters - i.e. values for which it can in **most** cases solve a task. The word **most** is important - notice that all a deep learning model is an `automatic mathematical optimization model` for a set of parameters. **`It solves tasks by approximation, not by building explicit logic`**.
- The process during which the model optimizes its parameters is called `training`.

</details>

<details>

<summary>How are deep learning models built?</summary>

Deep learning models are built by codifying the `description` of the desired model behavior. This description of the expected behavior is `implicitly` hidden in the data in the form of `patterns`. Thus, deep learning is uncovering those patterns and using them to solve various problems.

The process is called `training` - you give the untrained model your data (your `description` of desired behavior) and the model "tweaks" its parameters until it fits your description well enough. And there you have it - a deep learning model that does what you want (probably 😄 (i.e. with a certain probability, because it's never going to be perfect)).

You can think about the "tweaking" process as the process in which multiple models are created each with different values for their parameters, their accuracies are compared and the model with the highest accuracy is chosen as the final version.

</details>

## So what can deep learning models model?

Here is the twist - `everything`! For any task as long as you have enough data, you can model it.

One of the things you can model is **the probability of the next word in a sentence**. Surprise - the models that solve these types of tasks are called `Large Language Models`! You have a partially written sentence and you can create a mathematical model that predicts how likely every possible word, in the language you're working in, is to be the next word in that sentence. And after you've selected the word, you can repeat the process on this extended sentence - that's how you get `ChatGPT`.

## Modeling a neuron that can multiply by `2`

![w10_multiplier.png](assets/w10_multiplier.png "w10_multiplier.png")

We want to teach the model that `w` has to equal `2`.

<details>

<summary>How would we go about doing this?</summary>

1. We start with a random guess for `w`. For the sake of concreteness, let's say a random floating-point number in the interval `[0, 10)`. The interval does not matter  - even if it does not contain `2` the traning process would converge towards it.

![w10_multiplier_guess.png](assets/w10_multiplier_guess.png "w10_multiplier_guess.png")

2. We calculate the value of the loss function at that initial random guess.

![w10_multiplier_loss.png](assets/w10_multiplier_loss.png "w10_multiplier_loss.png")

3. We can see what will happen if we "wiggle" `w` by a tiny amount `eps`.

![w10_multiplier_loss_wiggle.png](assets/w10_multiplier_loss_wiggle.png "w10_multiplier_loss_wiggle.png")

4. So, the value either goes up or down. This means that our loss function would represent a parabola.

![w10_multiplier_loss_viz.png](assets/w10_multiplier_loss_viz.png "w10_multiplier_loss_viz.png")

5. If only we had a way to always know in which direction the value would go down? Oh, wait - we do! It's the opposite direction of the one in which the derivative grows!

![w10_derivative_values.gif](assets/w10_derivative_values.gif "w10_derivative_values.gif")

For now, we won't calculate the exact derivative because we don't need to do that - we can use its general formula:

$${\displaystyle L=\lim _{eps\to 0}{\frac {loss(w+eps)-loss(w)}{eps}}}$$

6. We can then use `L` to step in the direction of decline, by doing: `w -= L`.

7. This, however, will have a problem: the value of `L` might be very high. If our step is always `L` we would start oscilating. Therefore, we'll use a learning rate that will say how large our step would be: `w -= learning_rate * L`.

And this is it! This process is guaranteed to find `2` as the optimal value. Moreover, this iterative algorithm for minimizing a differentiable multivariate function is what is also known as [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) 😇.

</details>

<details>

<summary>What would the architecture and process for creating a model of an "AND" logical gate look like?</summary>

We might start off with something like this:

![w10_and_or_models.png](assets/w10_and_or_models.png "w10_and_or_models.png")

However, because our task now shifts from being a regression one into a classification one, we can also add the `sigmoid` function to control the output values:

$${\displaystyle f(x)={\frac {1}{1+e^{-x}}}}$$

![w10_and_or_models_sigmoid.png](assets/w10_and_or_models_sigmoid.png "w10_and_or_models_sigmoid.png")

<details>

<summary>But! Adding the sigmoid activation function actually causes another problem - for what values of w1 and w2 would we have a problem?</summary>

Look at what happens when we have $w_1=0$ and $w_2=0$ (our model is guessing correctly that the output should be `0`):

![w10_sigmoid_problem.png](assets/w10_sigmoid_problem.png "w10_sigmoid_problem.png")

</details>

<details>

<summary>How do we fix this?</summary>

We need to keep the weights at `0` but also add another term that can control the logit value when all weights are `0`. Welcome, ***bias***.

![w10_bias.png](assets/w10_bias.png "w10_bias.png")

</details>

</details>

<details>

<summary>How do we model the "XOR" logical gate?</summary>

Let's see how the classes are distributed in `2D` space:

![w10_class_distribution.png](assets/w10_class_distribution.png "w10_class_distribution.png")

The models we defined above are actually called perceptrons. They calculate a weighted sum of their inputs and thresholds it with a step function.

Geometrically, this means **the perceptron can separate its input space with a hyperplane**. That’s where the notion that a perceptron can only separate linearly separable problems comes from.

Since the `XOR` function **is not linearly separable**, it really is impossible for a single hyperplane to separate it.

<details>

<summary>What are our next steps then?</summary>

We need to describe the `XOR` gate using non-`XOR` gates. This can be done:

`(x|y) & ~(x&y)`

So, the `XOR` model can then be represented using the following architecture:

![w10_xor_architecture.png](assets/w10_xor_architecture.png "w10_xor_architecture.png")

<details>

<summary>How many parameters would we have in total?</summary>

9

</details>

</details>

</details>

# Week 11 - Implementing Gradient Descent

## Backpropagation

<details>

<summary>How do we translate the expression "slope of a line"?</summary>

Наклон на линия.

</details>

<details>

<summary>How would you define the slope of a line?</summary>

- slope (also gradient) = a number that describes the direction of the line on a plane.
- often denoted by the letter $m$.

![w11_slope.png](assets/w11_slope.png "w11_slope.png")

- calculated as the ratio of the vertical change to the horizontal change ("rise over run") between two distinct points on the line:
  - a 45° rising line has slope $m = 1$ (tan(45°) = 1)
  - a 45° falling line has slope $m = -1$ (tan(-45°) = -1)

</details>

<details>

<summary>What is the sign of the slope of an increasing line going up from left to right?</summary>

Positive ($m > 0$).

</details>

<details>

<summary>What is the sign of the slope of a decreasing line going down from left to right?</summary>

Negative ($m < 0$).

</details>

<details>

<summary>What is the slope of a horizontal line?</summary>

$0$.

</details>

<details>

<summary>What is the slope of a vertical line?</summary>

A vertical line would lead to a $0$ in the denominator, so the slope can be regarder as `undefined` or `infinite`.

</details>

<details>

<summary>What is the steepness of a line?</summary>

- The absolute value of its slope:
  - greater absolute value indicates a steeper line.

</details>

<details>

<summary>Suppose a line runs through two points: P = (1, 2) and Q = (13, 8). What is its slope, direction and level of steepness?</summary>

$dy = 8 - 2 = 6$
$dx = 13 - 1 = 12$
$m = \frac{dy}{dx} = \frac{6}{12} = \frac{1}{2} = 0.5$

Direction: $0.5 > 0$ => up
Steepness: $0 < 0.5 < 1$ => not very steep (less steep than a 45° rising line)

</details>

<details>

<summary>Suppose a line runs through two points: P = (4, 15) and Q = (3, 21). What is its slope, direction and level of steepness?</summary>

$dy = 21 - 15 = 6$
$dx = 3 - 4 = -1$
$m = \frac{dy}{dx} = \frac{6}{-1} = -6$

Direction: $-6 < 0$ => down
Steepness: $|-6| = 6 > 1$ => steep

</details>

<details>

<summary>What is the link between "slope" and "derivative"?</summary>

- For non-linear functions, the rate of change varies along the curve.
- The derivative of the function at a point
$=$ The slope of the line, tangent to the curve at the point
$=$ The rate of change of the function at that point

![w11_slop_der_connection.png](assets/w11_slop_der_connection.png "w11_slop_der_connection.png")

Formula for slope:

$m = \frac{dy}{dx}$

Formula for derivative:

${\displaystyle L=\lim _{eps\to 0}{\frac {f(x+eps)-f(x)}{eps}}}$

it's the same formula as for the slope, only here the change in $x$ is infinitesimally small.

For example, let $f$ be the squaring function: ${\displaystyle f(x)=x^{2}}$. Then the derivative is:

$$\frac{f(x+eps) - f(x)}{eps} = \frac{(x+eps)^2 - x^2}{eps} = \frac{x^2 + 2xeps + eps^2 - x^2}{eps} = 2x + eps$$

The division in the last step is valid as long as $eps \neq 0$. The closer $eps$ is to $0$, the closer this expression becomes to the value $2x$. The limit exists, and for every input $x$ the limit is $2x$. So, the derivative of the squaring function is the doubling function: ${\displaystyle f'(x)=2x}$.

</details>

<details>

<summary>So, what added value does the derivative have?</summary>

**It tells us by how much the value of a function increases when we *increase* its input by a tiny bit.**

Do we remember the below diagram?

![w10_multiplier_loss_viz.png](assets/w10_multiplier_loss_viz.png "w10_multiplier_loss_viz.png")

</details>

<details>

<summary>What are the rules of derivatives that you can recall - write out the rule and an example of it?</summary>

Recall the rules of computation [here](https://en.wikipedia.org/wiki/Derivative#Rules_of_computation).

Also, recall the chain rule [here](https://en.wikipedia.org/wiki/Chain_rule).

<details>

<summary>What is the derivative of sin(6x)?</summary>

$\frac{d}{dx}[\sin(6x)] = \cos(6x) * \frac{d}{dx}[6x] = \cos(6x) * 6 = 6\cos(x)$

See how the above corresponds with this definition:

$${\displaystyle {\frac {dz}{dx}}={\frac {dz}{dy}}\cdot {\frac {dy}{dx}},}$$

$z = \sin$
$y = 6x$

In other words, $x$ influences the value of $\sin$ through the value of $y=6x$.

</details>

</details>

<details>

<summary>What is backpropagation then?</summary>

Backpropagation is the iterative process of calculating derivatives of the loss function with respect to every `value` node leading up to it.

Rules of thumb:

```text
Start from the final child (the last node in topological order).
+ => copy gradient to parents:
    parent1.grad = current.grad
    parent2.grad = current.grad
* => multiply value of other parent with current gradient:
    parent1.grad = parent2.value * current.grad
    parent2.grad = parent1.value * current.grad
```

Let's say we have the following computational graph and we have to see how tiny changes in the weights and biases influence the value of `L`:

![w11_03_result](assets/w11_03_result.svg?raw=true "w11_03_result.png")

<details>

<summary>Reveal answer</summary>

![w11_calculations](assets/w11_calculations.png "w11_calculations.png")

End:

![w11_04_result](assets/w11_04_result.svg?raw=true "w11_04_result.png")

</details>

</details>

## Topological sort

Topological ordering of a directed graph is a linear ordering of its vertices such that for every directed edge $(u,v)$ from vertex $u$ to vertex $v$, $u$ comes before $v$ in the ordering.

The canonical application of topological sorting is in scheduling a sequence of jobs or tasks based on their dependencies.

Two ways to sort elements in topological order are given in [Wikipedia](https://en.wikipedia.org/wiki/Topological_sorting).

## The hyperbolic tangent

<details>

<summary>Why are activation functiosns needed?</summary>

They introduce nonlinearity, making it possible for our network to learn somthing. Composition of matrices is a single matrix (as the matrix is a linear operation).

</details>

$${\displaystyle \tanh x={\frac {\sinh x}{\cosh x}}={\frac {e^{x}-e^{-x}}{e^{x}+e^{-x}}}={\frac {e^{2x}-1}{e^{2x}+1}}.}$$

We observe that the `tanh` function is a shifted and stretched version of the `sigmoid`. Below, we can see its plot when the input is in the range $[-10, 10]$:

![w11_tanh](assets/w11_tanh.png "w11_tanh.png")

The output range of the tanh function is $(-1, 1)$ and presents a similar behavior with the `sigmoid` function. Thus, the main difference is the fact that the `tanh` function pushes the input values to $1$ and $-1$ instead of $1$ and $0$.

The important difference between the two functions is the behavior of their gradient.

$${\frac {d}{dx}}\sigma(x) = \sigma(x) (1 - \sigma(x))$$
$${\frac {d}{dx}}\tanh(x) = 1 - \tanh^{2}(x)$$

![w11_tanh_sigmoid_gradients](assets/w11_tanh_sigmoid_gradients.png "w11_tanh_sigmoid_gradients.png")

Using the `tanh` activation function results in higher gradient values during training and higher updates in the weights of the network. So, if we want strong gradients and big steps, we should use the `tanh` activation function.

Another difference is that the output of `tanh` is symmetric around zero, which could sometimes lead to faster convergence.

## Python OOP (Magic Methods)

### Initialization and Construction

- `__init__`: To get called by the `__new__` method. This is the `constructor` function for Python classes.
- `__new__`: To get called in an object’s instantiation (**do not use unless no other option**).
- `__del__`: It is the destructor (**do not use unless no other option**).

### Arithmetic operators

- `__add__(self, other)`: Implements behavior for the `+` operator (addition).
- `__sub__(self, other)`: Implements behavior for the `–` operator (subtraction).
- `__mul__(self, other)`: Implements behavior for the `*` operator (multiplication).
- `__floordiv__(self, other)`: Implements behavior for the `//` operator (floor division).
- `__truediv__(self, other)`: Implements behavior for the `/` operator (true division).
- `__mod__(self, other)`: Implements behavior for the `%` operator (modulus).
- `__pow__(self, other)`: Implements behavior for the `**` operator (exponentiation).
- `__and__(self, other)`: Implements behavior for the `&` operator (bitwise and).
- `__or__(self, other)`: Implements behavior for the `|` operator (bitwise or).
- `__xor__(self, other)`: Implements behavior for the `^` operator (bitwise xor).
- `__neg__(self)`: Implements behavior for negation using the `–` operator.

### String Magic Methods

- `__str__(self)`: Defines behavior for when `str()` is called on an instance of your class.
- `__repr__(self)`: To get called by built-int `repr()` method to return a machine readable representation of a type. **This method gets called when an object is passed to the `print` function.**

### Comparison magic methods

- `__eq__(self, other)`: Defines behavior for the equality operator, `==`.
- `__ne__(self, other)`: Defines behavior for the inequality operator, `!=`.
- `__lt__(self, other)`: Defines behavior for the less-than operator, `<`.
- `__gt__(self, other)`: Defines behavior for the greater-than operator, `>`.
- `__le__(self, other)`: Defines behavior for the less-than-or-equal-to operator, `<=`.
- `__ge__(self, other)`: Defines behavior for the greater-than-or-equal-to operator, `>=`.

# Week 12 - Hello, PyTorch

## PyTorch Dataset

<details>

<summary>What do you know about PyTorch?</summary>

[PyTorch](https://pytorch.org/docs/stable/index.html) is an optimized tensor library for deep learning using GPUs and CPUs.

- Open source.
- Created by Meta.
- It has everything needed to create a data preprocessing, model training and model evaluation pipeline.

We'll refer to any `n`-dimensional numpy array as a **tensor**.

![w12_tensor_meme](assets/w12_tensor_meme.png "w12_tensor_meme.png")

If you would like a more rigorous Math definition and are curious about the history of the term, feel free to read [this paper](https://arxiv.org/pdf/2510.12269).

</details>

This week we'll be working with `the water potability dataset`. The task is to classify a water sample as potable or drinkable (`1` or `0`) based on its chemical characteristics. All features have been normalized to between zero and one. Two files are present in our `DATA` folder: `water_train.csv` and `water_test.csv`. Here's how both of them look like:

![w12_water_potability_datasets.png](assets/w12_water_potability_datasets.png "w12_water_potability_datasets.png")

We can create a custom dataset for our data by inheriting the PyTorch [Dataset class](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). All customs `Dataset` classes must implement the following methods:

- `__init__`: to loads and saves the data in the state of the class. Typically accepts a CSV or an already loaded numpy matrix;
- `__len__`: returns the number of instaces in the saved data;
- `__getitem__`: returns the features and label for a single sample. Note: this method returns **a tuple**! The first element is an array of the features, the second is the label.

See an example [here](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files).

> **Note:** While it's not shown in the example, please don't forget to initialize the parent class as well by calling `super().__init__()`.

## PyTorch Dataloader

<details>

<summary>What is batch size?</summary>

A subsample of `n` instances. The batch size is used to split the dataset into multiple smaller dataset each with `n` instances. The benefit we get is having a smaller amount of data for one iteration of the training loop and faster training in general since the matrix calculations happen much faster.

</details>

The `Dataset` class retrieves our dataset’s features and labels one sample at a time.

While training a model, we typically want to:

- pass samples in `minibatches`;
- reshuffle the data at every epoch to reduce model overfitting;
- use Python’s `multiprocessing` package to speed up data retrieval.

The [`DataLoader` class](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) is an iterable that abstracts this complexity for us in an easy API.

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

We can then iterate through those objects via a `for` loop or manually:

```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze() # Removes dimensions of size 1. Shape (A×1×B×C×1×D) becomes (A×B×C×D).
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

## PyTorch Model

- We can define neural networks in two ways: `sequential` and `class-based`. We'll focus on the `class-based` approach as it provides more flexibility.
- All functions and methods that work with neural networks can be found in the [torch.nn package](https://pytorch.org/docs/stable/nn.html). By convention `torch.nn` gets imported with an alias `nn`.

Here's an example of sequential model definition:

```python
import torch.nn as nn

net = nn.Sequential(
  nn.Linear(9, 16),
  nn.ReLU(),
  nn.Linear(16, 8),
  nn.ReLU(),
  nn.Linear(8, 1),
  nn.Sigmoid(),
)
```

The same can be written using the `class-based` approach:

```python
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(9, 16)
    self.fc2 = nn.Linear(16, 8)
    self.fc3 = nn.Linear(8, 1)
  
  def forward(self, x):
    x = nn.functional.relu(self.fc1(x))
    x = nn.functional.relu(self.fc2(x))
    x = nn.functional.sigmoid(self.fc3(x))
    return x

net = Net()
```

As can be seen above, every model should define the following two methods:

- `__init__()`: defines the layers that are used in the `forward()` method;
- `forward()`: defines what happens to the model inputs once it receives them; this is where you pass inputs through pre-defined layers.

By convention `torch.nn.functional` gets imported with an alias `F`. That means that the above body of `forward` can be rewritten like:

```python
import torch.nn.functional as F

...

x = F.relu(self.fc1(x))
x = F.relu(self.fc2(x))
x = F.sigmoid(self.fc3(x))
```

- PyTorch has many famous deep learning models already built-in.
- For example, various vision models can be found in the [torchvision.models package](https://pytorch.org/vision/0.9/models.html).

## Training loop - PyTorch style

Here is a general overview of the training loop using built-in PyTorch functions and classes.

```python
import torch.nn as nn
import torch.optim as optim

criterion = nn.BCELoss() # we define the loss function to use. BCE = Binary Cross Entropy
optimizer = optim.SGD(net.parameters(), lr=0.01) # we set the optimizer. SGD = Stochastic Gradient Descent

for epoch in range(1000):
  for features, labels in dataloader_train:
    optimizer.zero_grad() # clear the gradients
    outputs = net(features) # forward pass
    loss = criterion(outputs, labels.view(-1, 1)) # calculate the loss
    loss.backward() # compute the gradients
    optimizer.step() # tweak weights
```

## Optimizers

- Stochastic Gradient Descent (SGD):
  - Parameter update values depend on learning rate;
  - Simple and efficient for basic models.
  - Rarely used in practice, due to simplicity.

- Adaptive Gradient (Adagrad):
  - Adapts learning rate for each parameter;
  - Good for sparse data;
  - May decrease the learning rate too fast.

- Root Mean Square Propagation (RMSprop):
  - Update for each parameter based on the size of its previous gradients.

- Adaptive Moment Estimation (Adam):
  - Has an extension with weight decay, called `AdamW`, which is arguably the most versatile and widely used optimizer today.
  - RMSprop + gradient momentum.
  - Often `Adam` is used as the go-to optimizer.

## Model evaluation - PyTorch style

```python
from torchmetrics import Accuracy

acc = Accuracy(task='binary')

net.eval()
with torch.no_grad():
  for features, labels in dataloader_test:
    outputs = net(features)
    preds = (outputs >= 0.5).float()
    acc(preds, labels.view(-1, 1))

accuracy = acc.compute()
print(f'Accuracy" {accuracy}')
```

### Unstable gradients

- **Vanishing gradients**: Gradients get smaller and smaller during backward pass.

![w12_vanishing_gradients.png](./assets/w12_vanishing_gradients.png "w12_vanishing_gradients.png")

- Results:
  - Earlier layers get smaller parameter updates;
  - Model does not learn.
  - Loss becomes constant.

- **Exploding gradients**: Gradients get larger and larger during backward pass.

![w12_exploding_gradients.png](./assets/w12_exploding_gradients.png "w12_exploding_gradients.png")

- Results:
  - Parameter updates are too large.
  - Loss becomes higher and higher.

### Solutions to unstable gradients

1. Proper weights initialization.
2. More appropriate activation functions.
3. Batch normalization.

#### Proper weights initialization

Good weight initialization ensures that the:

- Variance of layer inputs = variance of layer outputs;
- Variance of gradients is the same before and after a layer.

How to achieve this depends on the activation function:

- For ReLU and similar (sigmoid included), we can use [He/Kaiming initialization](https://paperswithcode.com/method/he-initialization).

```python
import torch.nn.init as init

init.kaiming_uniform_(layer.weight) # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
```

#### More appropriate activation functions

- Often the `ReLU` (rectified linear unit) activation function is used.

![w12_relu.png](./assets/w12_relu.png "w12_relu.png")

<details>

<summary>ReLU is not differentiable at x=0. So, how do we implement backpropagation?</summary>

For $x > 0$, the derivative of the ReLU function is $1$. For $x <= 0$, the derivative of the ReLU function is $0$. If $x == 0$, it's usually set either to $0$ or to a very small $eps$.

</details>

- It's available in `nn.functional.relu()`.
- It suffers from the `dying neuron` problem because it's `0` for negative inputs.

- Alternative of `ReLU` is `ELU` (Exponential Linear Unit ):

![w12_elu.png](./assets/w12_elu.png "w12_elu.png")

- Available in `nn.functional.elu()`.
- Non-zero gradients for negative values - helps against dying neurons.
- Average output is around `0` - helps against vanishing gradients.

#### Batch normalization

- Good choice of initial weights and activations doesn't prevent unstable gradients during training (only during initialization).
- Solution is to add another transformation after each layer - batch normalization:
  1. Standardizes the layer's outputs by subtracting the mean and diving by the standard deviation.
  2. Scales and shifts the standardized outputs using learnable parameters.
- Result:
  - Model learns optimal distribution of inputs for each layer.
  - Faster loss decrease.
  - Helps against unstable gradients during training.
- Available as [`nn.BatchNorm1d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html).
  - **Note 1:** The number of features has to be equal to the number of output neurons of the previous layer.
  - **Note 2:** Done after applying layer and before the activation.

# Week 13 - Convolutional Neural Networks

## The Clouds dataset

We will be working with a dataset containing pictures of various types of clouds.

![w13_task01.png](assets/w13_task01.png "w13_task01.png")

<details>

<summary>How can we load one of those images in Python?</summary>

We can use the [`pillow`](https://pypi.org/project/pillow/) package. It is imported with the name `PIL` and has a very handly [`Image.open` function](https://pillow.readthedocs.io/en/latest/handbook/tutorial.html#using-the-image-class).

</details>

<details>

<summary>Wait - what is an image again?</summary>

- The image is a matrix of pixels ("picture elements").
- Each pixel contains color information.

![w13_image_pixels.png](assets/w13_image_pixels.png "w13_image_pixels.png")

- Grayscale images: integer in the range $[0 - 255]$.
  - 30:

    ![w13_image_gray.png](assets/w13_image_gray.png "w13_image_gray.png")

- Color images: three/four integers, one for each color channel (**R**ed, **G**reen, **B**lue, sometimes also **A**lpha).
  - RGB = $(52, 171, 235)$:

    ![w13_image_blue.png](assets/w13_image_blue.png "w13_image_blue.png")

</details>

## Converting pixels to tensors and tensors to pixels

### [`ToTensor()`](https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html#totensor)

- Converts pixels to float tensors (PIL image => `torch.float`).
- Scales values to $[0.0, 1.0]$.

### [`PILToTensor()`](https://pytorch.org/vision/main/generated/torchvision.transforms.PILToTensor#piltotensor)

- Converts pixels to `8`-bit unsigned integers (PIL image => `torch.uint8`).
- Does not scale values: they stay in the interval $[0, 255]$.

### [`ToPILImage()`](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.ToPILImage)

- Converts a tensor or an numpy `ndarray` to PIL Image.
- Does not change values.

## Loading images to PyTorch

The easiest way to build a `Dataset` object when we have a classification task is with a predefined directory structure.

As we load the images, we could also apply preprocessing steps using `torchvision.transforms`.

```text
clouds_train
  - cumulus
    - 75cbf18.jpg
    - ...
  - cumulonimbus
  - ...
clouds_test
  - cumulus
  - cumulonimbus
```

- Main folders: `clouds_train` and `clouds_test`.
  - Inside: one folder per category.
    - Inside: image files.

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data

train_transforms = transforms.Compose([
  transforms.ToTensor(), # convert the object into a tensor
  transforms.Resize((128, 128)), # resize the images to be of size 128x128
])

dataset_train = ImageFolder(
  'DATA/clouds/clouds_train',
  transform=train_transforms,
)

dataloader_train = data.DataLoader(
  dataset_train,
  shuffle=True,
  batch_size=1,
)

image, label = next(iter(dataloader_train))
print(image.shape)
```

```console
torch.Size([1, 3, 128, 128])
```

In the above output:

- `1`: batch size;
- `3`: three color channels;
- `128`: height;
- `128`: width.

We could display these images as well, but we'll have to do two transformations:

1. We need to have a three dimensional matrix. The above shape represents a `4D` one. To remove all dimensions with size `1`, we can use the `squeeze` method of the `image` object.
2. The number of color channels must come after the height and the width. To change the order of the dimensions, we can use the `permute` method of the `image` object.

```python
image = image.squeeze().permute(1, 2, 0)
print(image.shape)
```

```console
torch.Size([128, 128, 3])
```

We can now, plot this using `matplotlib`:

```python
import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis('off')
plt.show()
```

![w13_loading_image_result.png](assets/w13_loading_image_result.png "w13_loading_image_result.png")

## Data augmentation

<details>

<summary>What is data augmentation?</summary>

Applying random transformations to original data points.

</details>

<details>

<summary>What is the goal of data augmentation?</summary>

Generating more data.

</details>

<details>

<summary>On which set should data augmentation be applied to - train, validation, test, all, some?</summary>

Only to the training set.

</details>

<details>

<summary>What is the added value of data augmentation?</summary>

- Increase the size of the training set.
- Increase the diversity of the training set.
- Improve model robustness.
- Reduce overfitting.

</details>

All supported image augmentation transformation can be found in [the documentation of torchvision](https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended).

<details>

<summary>Image augmentation operations can sometimes negatively impact the training process. Can you think of two deep learning tasks in which specific image augmentation operations should not be used?</summary>

- Fruit classification and changing colors:

One of the supported image augmentation transformations is [`ColorJitter`](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html#colorjitter) - it randomly changes brightness, contrast, saturation, hue, and other properties of an image.

If we are doing fruit classification and decide to apply a color shift augmentation to an image of the lemon, the augmented image will still be labeled as lemon although it would represent a lime.

![w13_data_augmentation_problem.png](assets/w13_data_augmentation_problem.png "w13_data_augmentation_problem.png")

- Hand-written characters classification and vertical flip:

![w13_data_augmentation_problem2.png](assets/w13_data_augmentation_problem2.png "w13_data_augmentation_problem2.png")

</details>

<details>

<summary>So, how do we choose appropriate augmentation operations?</summary>

- Whether an augmentation operation is appropriate depends on the task and data.
- Remember: Augmentations impact model performance.

<details>

<summary>Ok, but then how do we see the dependence in terms of data?</summary>

Explore, explore, explore!

</details>

</details>

<details>

<summary>What transformations can you think of for our current task (cloud classification)?</summary>

```python
train_transforms = transforms.Compose([
  transforms.RandomHorizontalFlip(), # simulate different viewpoints of the sky
  transforms.RandomRotation(45), # expose model to different angles of cloud formations
  transforms.RandomAutocontrast(), # simulate different lighting conditions
  transforms.ToTensor(), # convert the object into a tensor
  transforms.Resize((128, 128)), # resize the images to be of size 128x128
])
```

![w13_data_augmentation.png](assets/w13_data_augmentation.png "w13_data_augmentation.png")

</details>

Which of the following statements correctly describe data augmentation? (multiple selection)

A. Using data augmentation allows the model to learn from more examples.
B. Using data augmentation increases the diversity of the training data.
C. Data augmentation makes the model more robust to variations and distortions commonly found in real-world images.
D. Data augmentation reduces the risk of overfitting as the model learns to ignore the random transformations.
E. Data augmentation introduces new information to the model that is not present in the original dataset, improving its learning capability.
F. None of the above.

<details>

<summary>Reveal answer</summary>

Answers: A, B, C, D.

Data augmentation allows the model to learn from more examples of larger diversity, making it robust to real-world distortions.

It tends to improve the model's performance, but it does not create more information than is already contained in the original images.

<details>

<summary>What should we prefer - using more real training data or generating it artificially?</summary>

If available, using more training data is preferred to creating it artificially with data augmentation.

</details>

</details>

## CNNs - The neural networks for image processing

Let's say that we have the following image:

![w13_linear_layers_problem.png](assets/w13_linear_layers_problem.png "w13_linear_layers_problem.png")

<details>

<summary>What is the problem of using linear layers to solve the classification task?</summary>

Too many parameters.

If the input size is `256x256`, that means that the network has `65,536` inputs!

If the first linear layer has `1000` neurons, only it alone would result in over `65` **million** parameters! For a color image, this number would be even higher.

![w13_linear_layers_problem2.png](assets/w13_linear_layers_problem2.png "w13_linear_layers_problem2.png")

So, the three main problems are:

- Incredible amount of resources needed.
- Slow training.
- Overfitting.

</details>

<details>

<summary>What is another more subtle problem of using linear layers only?</summary>

They are not space-invariant.

Linearly connected neurons could learn to detect the cat, but the same cat **won't be recognized if it appears in a *different* location**.

![w13_space_invariance.png](assets/w13_space_invariance.png "w13_space_invariance.png")

</details>

<details>

<summary>So, ok - the alternative is using CNNs. How do they work?</summary>

![w13_cnn.png](assets/w13_cnn.png "w13_cnn.png")

- Parameters are collected in one or more small grids called **filters**.
- **Slide** filter(s) over the input.
- At each step, perform the **convolution** operation.
- The end result (after the whole image has been traversed with **one filter**) is called a **feature map**:
  - Preserves spatial patterns from the input.
  - Uses fewer parameters than linear layer.
- Remember: one filter = one feature map. We can slide **multiple filters over an original image**, to get multiple feature maps.
- We can then apply activations to the feature maps.
- The set of all feature maps combined, form the output of a single convolutional layer.
- Available in `torch.nn`: [`nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html):
  - `in_channels`: number of input dimensions.
  - `out_channels`: number of filters.
  - `kernel_size`: height and width of the filter(s).

</details>

<details>

<summary>What does the convolution operation comprise of?</summary>

![w13_cnn_dot.png](assets/w13_cnn_dot.png "w13_cnn_dot.png")

</details>

<details>

<summary>What is zero padding?</summary>

![w13_cnn_zero_pad.png](assets/w13_cnn_zero_pad.png "w13_cnn_zero_pad.png")

- Add a frame of zeros to the input of the convolutional layer.
- This maintains the spatial dimensions of the input and output tensors.
- Ensures border pixels are treated equally to others.

Available as `padding` argument: `nn.Conv2d(3, 32, kernel_size=3, padding=1)`.

</details>

<details>

<summary>What is max pooling?</summary>

![w13_cnn_max_pool.png](assets/w13_cnn_max_pool.png "w13_cnn_max_pool.png")

- Slide non-overlapping window over input.
- At each position, retain only the maximum value.
- Used after convolutional layers to reduce spatial dimensions.

Available in `torch.nn`: [`nn.MaxPool2d(kernel_size=2)`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html).

</details>

<details>

<summary>But what happens when the input is three dimensional - what do the filters look like?</summary>

The `2D` part of a `2D` convolution does not refer to the dimension of the convolution input, nor of dimension of the filter itself, but rather of the space in which the filter is allowed to move (`2` directions only).

Different `2D`-array filters are applied to each dimension and then their outputs are summed up (you can also think of this as a single `3D` matrix):

![w13_rgb_convolution.png](assets/w13_rgb_convolution.png "w13_rgb_convolution.png")

$$(1*1+2*2+4*3+5*4)+(0*0+1*1+3*2+4*3) = 56$$

More on this [here](https://stackoverflow.com/a/62544803/16956119) and [here](https://d2l.ai/chapter_convolutional-neural-networks/channels.html).

</details>

## Architecture

The typical architecture follows the style:

1. Convolution.
2. Activation function - `ReLU`, `ELU`, etc.
3. Max pooling.
4. Iterate the above until there are much more filters than height and width (effectively, until we get a much "deeper" `z`-axis).
5. Flatter everything into a single vector using [`nn.Flatten`](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html). This vector is the ***summary of the original input image***.
6. Apply several regular linear layers to it.

![w13_architecture2.png](assets/w13_architecture2.png "w13_architecture2.png")

Here's one famous architecture - [VGG-16](https://arxiv.org/abs/1409.1556v6):

![w13_architecture.png](assets/w13_architecture.png "w13_architecture.png")

In our case, we could have something like the following:

![w13_architecture_ours.png](assets/w13_architecture_ours.png "w13_architecture_ours.png")

Which of the following statements are true about convolutional layers? (multiple selection)

A. Convolutional layers preserve spatial information between their inputs and outputs.
B. Adding zero-padding around the convolutional layer's input ensures that the pixels at the border receive as much attention as those located elsewhere in the feature map.
C. Convolutional layers in general use fewer parameters than linear layers.

<details>

<summary>Reveal answer</summary>

All of them.

</details>

## Cross-Entropy loss

$${\displaystyle H(model,labels)=-\sum _{x\in {\mathcal {X}}}model(x)\,\log labels(x).}$$

- Available in [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss).
- Can apply it directly on logits: it includes the `softmax` logic.

## Checkpoint

You are building a model to recognize different flower species. You consider three augmentations to use: `color shift`, `rotation`, and `texture change`. The graphic below shows the examples of these three augmentations. Which of the augmentations could be used for the flower classifier?

![w13_checkpoint.png](assets/w13_checkpoint.png "w13_checkpoint.png")

A. Only color shift
B. Only rotation
C. Only texture shift
D. All three
E. None of the three

<details>

<summary>Reveal answer</summary>

Answer: B.

The flower is still the same species when it's rotated, so this augmentation will not introduce noise in training labels; rather, it will increase the diversity of the training samples.

For this task having the right color is important. The other two operations can change the color which could negatively impact the performance of the model.

</details>

## Precision & Recall for Multiclass Classification (revisited)

### Computing total value

In week 04 we introduced the metrics `precision` and `recall` in the context of binary classification.

We also touched very slightly on the usecase for multiclass classification:

```console
              precision    recall  f1-score   support

     class 0       0.50      1.00      0.67         1
     class 1       0.00      0.00      0.00         1
     class 2       1.00      0.67      0.80         3

    accuracy                           0.60         5
   macro avg       0.50      0.56      0.49         5
weighted avg       0.70      0.60      0.61         5
```

`Support` represents the number of instances for each class within the true labels. If the column with `support` has different numbers, then we have class imbalance.

- `macro average` = $\frac{F1_{class1} + F1_{class2} + F1_{class3}}{3}$
- `weighted average` = $\frac{F1_{class1}*SUPPORT_{class1} + F1_{class2}*SUPPORT_{class2} + F1_{class3}*SUPPORT_{class3}}{3}$
- `micro average` = $\frac{F1_{class1}*SUPPORT_{class1} + F1_{class2}*SUPPORT_{class2} + F1_{class3}*SUPPORT_{class3}}{SUPPORT_{class1} + SUPPORT_{class2} + SUPPORT_{class3}}$

To calculate them with `torch`, we can use the classes [`torchmetrics.Recall`](https://lightning.ai/docs/torchmetrics/stable/classification/recall.html) and [`torchmetrics.Precision`](https://lightning.ai/docs/torchmetrics/stable/classification/precision.html):

```python
from torchmetrics import Recall

recall_per_class = Recall(task='multiclass', num_classes=7, average=None)
recall_micro = Recall(task='multiclass', num_classes=7, average='micro')
recall_macro = Recall(task='multiclass', num_classes=7, average='macro')
recall_weighted = Recall(task='multiclass', num_classes=7, average='weighted')
```

When to use each:

- `micro`: imbalanced datasets.
- `macro`: consider errors in small classes as equally important as those in larger classes.
- `weighted`: consider errors in larger classes as most important.

We also have the [F1 score metric](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html).

<details>

<summary>What does multilabel classification mean?</summary>

When one instance can get multiple classes assigned to it. This is the case, for example, in research article authorship identification: one article has multiple authors.

</details>

### Computing per class value

- We can also analyze the performance per class.
- To do this, compute the metric, setting `average=None`.
  - This gives one score per each class:

```python
print(f1)
```

```console
tensor([0.6364, 1.0000, 0.9091, 0.7917,
        0.5049, 0.9500, 0.5493],
        dtype=torch.float32)
```

- Then, use the `Dataset`'s `.class_to_idx` attribute that maps class names to indices.

```python
dataset_test.class_to_idx
```

```console
{'cirriform clouds': 0,
 'clear sky': 1,
 'cumulonimbus clouds': 2,
 'cumulus clouds': 3,
 'high cumuliform clouds': 4,
 'stratiform clouds': 5,
 'stratocumulus clouds': 6}
```

# Week 14 - Recurrent Neural Networks

## Sequential data

<details>

<summary>What is sequential data?</summary>

- **Ordered** in time or space.
- Having an order between the data points, means that there is an **implicit dependency** between them.

</details>

<details>

<summary>Give three examples of sequential data.</summary>

- Text.
- Time series (stock market movements).

![w14_sequential_data_examples.png](assets/w14_sequential_data_examples.png "w14_sequential_data_examples.png")

- Audio waves.

![w14_audio_waves.png](assets/w14_audio_waves.png "w14_audio_waves.png")

</details>

<details>

<summary>What were the typical preprocessing steps we talked about?</summary>

- Analyzing the data type.
- Data statistics.
- Data distribution.
- Correlation analysis for feature elimination.
- Train-test split.
- Data scaling.
- Missing value imputation or removal.
- One-hot encoding.

</details>

<details>

<summary>From the above which is not applicable when solving a problem involving sequential data?</summary>

No random train-test splitting when working with sequential data!

<details>

<summary>Why?</summary>

Introduces the problem of **look-ahead** bias in which the model has information about the future.

</details>

</details>

<details>

<summary>How can we solve this?</summary>

1. Split by time, not by observation.

![w14_split_by_time_example.png](assets/w14_split_by_time_example.png "w14_split_by_time_example.png")

- Pros:
  - Straightforward.
  - Great for simple use-cases.
- Cons:
  - Model becomes unreliable as OOT samples grow.

2. Use a rolling window approach: take $n$ samples and predict the $n + 1$. **<- preferred**

</details>

## Electricity consumption prediction

Let's tackle the problem of predicting electricity consumption based on past patterns. In our `DATA` folder you'll find the folder `electricity_consumption`. It contains electricity consumption in kilowatts, or kW, for a certain user recorded every 15 minutes for four years.

![w14_dataset_example.png](assets/w14_dataset_example.png "w14_dataset_example.png")

<details>

<summary>What would a single training example even look like?</summary>

- Our models will now take multiple rows of the data and try to predict the row that follows.
- In other words, to feed the training data to the model, we need to chunk it first to create **sequences** that the model can use as training examples.

</details>

<details>

<summary>Having understood this, what would the term "sequence length" mean?</summary>

The number of features in one training example.

![w14_seq_len_example.png](assets/w14_seq_len_example.png "w14_seq_len_example.png")

</details>

<details>

<summary>If we wanted to base our prediction on the last 24 hours of consumption, what would be the sequence length of a single training example?</summary>

Because the measurements are per $15$ minutes, $4$ measurements correspond to $1$ hour. To get $24$ hours, we would take $24 * 4 = 96$ samples.

</details>

The above data-splitting techniques are applicable to other types of sequential data as well:

- Large Language Models: predict the next word in a sentence (similar to predicting the next amount of electricity used);
- Speech recognition: transcribing an audio recording of someone speaking to text.

## [`TensorDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset)

If we have `numpy` arrays of training and testing data and want to use them as a `pytorch` `Dataset`, we can use the `TensorDataset` wrapper.

```python
X_train, y_train = create_sequences(train_data, seq_length)
print(X_train.shape, y_train.shape)
```

```console
(34944, 96) (34944,)
```

```python
from torch.utils.data import TensorDataset

dataset_train = TensorDataset(
  torch.from_numpy(X_train).float(),
  torch.from_numpy(y_train).float(),
)
```

The `TensorDataset` behaves just like all other torch `Datasets` and it can be passed to a `DataLoader` in the same way.

## The recurrent neuron

- So far, we've only seen feed-forward networks.
- RNNs are a different class of networks that have connections pointing back at the same neuron.
- Recurrent neuron:
  - Input `x`.
  - Output `y`.
  - Hidden state `h`.
- In PyTorch: `nn.RNN()`.

![w14_rnn_example.png](assets/w14_rnn_example.png "w14_rnn_example.png")

#### Internals of [the `RNN` cell](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)

![w14_rnn_internals.png](assets/w14_rnn_internals.png "w14_rnn_internals.png")

- Two inputs:
  - current input data `x`;
  - previous hidden state `h`.
- Two outputs:
  - current output `y`;
  - next hidden state `h`.
- What happens in the block `weights & activation` is subject to implementation - **different styles exist**:

1. We can weigh the input and the hidden state and then sum them up (PyTorch style):

```python
class RNNCell:
  # ...
  def step(self, x):
    # update the hidden state
    self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
    # compute the output vector
    y = np.dot(self.W_hy, self.h)
    return y, self.h
```

2. We can also just concatenate them and pass them through a `nn.Linear` layer:

![w14_rnn_internals_v2.gif](assets/w14_rnn_internals_v2.gif "w14_rnn_internals_v2.gif")

#### Unrolling through time

- The output at each time step depends on all the previous inputs.
- Recurrent networks maintain **memory through time**, which allows them to handle sequential data well.

![w14_unrolling_rnn.png](assets/w14_unrolling_rnn.png "w14_unrolling_rnn.png")

![w14_unrolling_rnn2.gif](assets/w14_unrolling_rnn2.gif "w14_unrolling_rnn2.gif")

#### Deep RNNs

![w14_deep_rnn.png](assets/w14_deep_rnn.png "w14_deep_rnn.png")

### Architecture types

- Depending on the lengths of input and output sequences, we distinguish four different architecture types:
  - `sequence-to-sequence` (`seq2seq`);
  - `sequence-to-vector`;
  - `vector-to-sequence`;
  - `encoder-decoder`.

#### Sequence-to-sequence architecture

Pass sequence as input, use entire output sequence.

![w14_seq2seq.png](assets/w14_seq2seq.png "w14_seq2seq.png")

#### Sequence-to-vector architecture

Pass sequence as input, use only the last output.

![w14_seq2vec.png](assets/w14_seq2vec.png "w14_seq2vec.png")

In PyTorch:

```python
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = nn.RNN(
        input_size=1,     # Number of features in the input at each time step
        hidden_size=32,   # Size of the hidden state (memory vector)
        num_layers=2,     # Two stacked RNN layers
        batch_first=True  # When `True` the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
    )
    self.fc = nn.Linear(32, 1)

  def forward(self, x):
    # Initialize the hidden state with zeros
    # Shape: (num_layers, batch_size, hidden_size)
    h0 = torch.zeros(2, x.size(0), 32, device=x.device)
    # x shape: (batch_size, sequence_length, input_size)
    out, h_n = self.rnn(x, h0)
    # out shape: (batch_size, sequence_length, hidden_size)
    # h_n shape: (num_layers, batch_size, hidden_size)
    # Use the hidden state from the last time step (for a standard (non-bidirectional) RNN: out[:, -1, :] === h_n[-1, :, :])
    out = self.fc(out[:, -1, :]) # (batch_size, hidden_size)
    return out
```

#### Vector-to-sequence architecture

Pass single input, use the entire output sequence.

![w14_vec2seq.png](assets/w14_vec2seq.png "w14_vec2seq.png")

#### Encoder-decoder architecture

Pass entire input sequence, only then start using output sequence.

![w14_enc_dec.png](assets/w14_enc_dec.png "w14_enc_dec.png")

#### Choosing the right architecture for the problem

Whenever you face a task that requires handling sequential data, you need to be able to decide what type of recurrent architecture is the most suitable for the job. Classify each of the provided use cases to the most suitable recurrent neural network architecture.

![w14_all_architectures.png](assets/w14_all_architectures.png "w14_all_architectures.png")

<details>

<summary>Based on a sequence of words, predict whether the sentiment is positive or negative.</summary>

Sequence-to-vector.

</details>

<details>

<summary>Translate text from Bulgarian to Greek.</summary>

Encoder-decoder.

</details>

<details>

<summary>Real-time speech recognition.</summary>

Sequence-to-sequence.

</details>

<details>

<summary>Non-real time speech recognition when the device first listens, and only then replies.</summary>

Encoder-decoder.

</details>

<details>

<summary>Image captioning: pass a representation of an image and get a caption for it.</summary>

Vector-to-sequence.

</details>

<details>

<summary>Based on the stock price in the last 30 days, predict the stock price today.</summary>

Sequence-to-vector.

</details>

<details>

<summary>Given a topic, generate text on this topic.</summary>

Vector-to-sequence.

</details>

<details>

<summary>Video classification on a frame level.</summary>

Sequence-to-sequence.

</details>

### The problem with using plain RNNs the model sequential data

<details>

<summary>What do you think?</summary>

- RNN cells maintain memory via the hidden state.
- This memory, however, is very short-lived: plain RNNs suffer from the **short-term memory problem**.
- Two more powerful cells solve this problem:
  - The Long Short-Term Memory (`LSTM`) cell.
  - The Gated Recurrent Unit (`GRU`) cell.

</details>

![w14_unstable_gradient.gif](assets/w14_unstable_gradient.gif "w14_unstable_gradient.gif")

### Internals of [the `LSTM` cell](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

![w14_lstm_internals.png](assets/w14_lstm_internals.png "w14_lstm_internals.png")

- Two hidden states:
  - `h`: short-term state;
  - `c`: long-term state.
- Three inputs and outputs:
  - current input `x`;
  - previous short-term state;
  - previous long-term state.
- Three `gates`:
  - **Forget gate:** what to remove from long-term memory.
  - **Input gate:** what to save to long-term memory.
  - **Output gate:** what to return at the current time step.

![w14_lstm_internals2.png](assets/w14_lstm_internals2.png "w14_lstm_internals2.png")

Example:

```python
rnn = nn.LSTM(10, 20, 2)
input_x = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input_x, (h0, c0))
```

![w14_lstm_internals3.gif](assets/w14_lstm_internals3.gif "w14_lstm_internals3.gif")

### Internals of [the `GRU` cell](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)

![w14_gru_internals.png](assets/w14_gru_internals.png "w14_gru_internals.png")

- Simplified version of the `LSTM` cell.
- Just one hidden state.
- No output gate.

![w14_gru_internals2.png](assets/w14_gru_internals2.png "w14_gru_internals2.png")

Example:

```python
rnn = nn.GRU(10, 20, 2)
input_x = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input_x, h0)
```

![w14_gru_internals3.gif](assets/w14_gru_internals3.gif "w14_gru_internals3.gif")

### Should I use RNN, LSTM, or GRU?

- Experiment, experiment, experiment.
- General trends:
  - RNN is not used a lot.
  - GRU is simpler than LSTM. It has less computation and is therefore faster.
  - Relative performance varies per use-case. Use the simplest model that achieves the metrics you want.

Which of the following statements about the different types of recurrent neural networks are correct? (multiple selection)

A. Plain RNN cells are able to keep a memory track of long input sequences.
B. LSTM cells keep two hidden states: one for short-term memory and one for long-term memory.
C. In both LSTM and GRU cells, the hidden state is not only passed as input to the next time step, but also returned (either as a copy or through an activation function) at the current time step.
D. LSTM cells always provide better results than GRU cells because they can handle more complexity with three rather than two gates.

<details>

<summary>Reveal answer</summary>

Answer: B, C.

</details>

# Week 15 - Bigram language model using counting

## [`PyCaret`](https://pycaret.org/): Low-code Machine Learning

We've seen that ML workflows often involve multiple steps: data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation.

Traditionally, these steps require significant coding effort and expertise in libraries like scikit-learn, XGBoost, or PyTorch. For beginners or those who want rapid prototyping, this complexity can be overwhelming.

PyCaret addresses these challenges by providing a **low-code, automated machine learning framework**. It allows you to do automatic preprocessing, exploration, modelling, evaluation and finetuning: <https://pycaret.gitbook.io/docs>.

See the examples here:
  - <https://pycaret.gitbook.io/docs/get-started/functions/train#example>
  - <https://nbviewer.org/github/pycaret/examples/tree/main/>

Note there are some limitations:

- Less flexibility: You lose control on how elements are applied which means that you can't build custom pipelines.
- Not ideal for production-scale ML: I suggest you use it for rapid prototyping and small-to-medium projects rather than large-scale systems as they typically require more experiments and careful tweaking of the hyperparameters.

Ok - let's now get to the main session!

## Generating human names

Our goal for this final session would be to create a statistical model that can generate human-like names. More precisely, we'll develop a character-level bigram language model built just from counting (no fancy neural networks and backpropagation).

## Character-level language model

<details>
<summary>What is a language model?</summary>

An algorithm that returns probabilities for the possible next token in a sequence.

</details>

<details>
<summary>What is a character-level language model?</summary>

- When the tokens are characters.
- The language model is therefore:
  - modelling sequences of characters;
  - able to predict the next character in a sequence of characters.

</details>

## An $n$-gram language model

<details>
<summary>What do you know about these types of language models?</summary>

$n$-gram language models are statistical models over sequences of tokens:

- if the tokens are words, then we have a [word $n$-gram language model](https://en.wikipedia.org/wiki/Word_n-gram_language_model);
- if they are characters, then we have a character-level language model.

</details>

<details>
<summary>What does "n" refer to?</summary>

$n$ = length of the sequence = size of the context.

</details>

<details>
<summary>Can a token be a sequence of characters (but not a full word) or a number of words (but not a full sentence)?</summary>

Yep - actually, such methods are used in GPT-3.5 and GPT-4. Feel free to read more about [Byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding).

</details>

<details>
<summary>What is the assumption that drives the logic of such models?</summary>

The probability of the next token in a sequence depends only on a fixed size window of previous tokens.

</details>

<details>
<summary>What is then a bigram model?</summary>

When only one previous token is considered to predict the current/next one.

</details>

<details>
<summary>What is then a trigram model?</summary>

When two previous tokens are considered to predict the current/next one.

</details>

<details>
<summary>And so, what is then an n-gram model?</summary>

When $n - 1$ previous tokens are considered to predict the current/next one.

</details>

## Building out a character-level bigram language model

The first thing we think about when creating a model is the data. Here are the first few names in our training dataset:

```text
emma
olivia
ava
isabella
```

<details>
<summary>What would the training examples for our model look like?</summary>

- Every name has quite a few examples packed in it.
- An existence of a word like `isabella` is telling us that:
  1. The character `i` is likely to be followed by the character `s`.
  2. The character `s` is likely to be followed by the character `a`.
  3. The character `a` is likely to be followed by the character `b`.
  4. The character `b` is likely to be followed by the character `e`.
  5. The character `e` is likely to be followed by the character `l`.
  6. The character `l` is likely to be followed by the character `l`.
  7. The character `l` is also likely to be followed by the character `a`.

</details>

<details>
<summary>However, we have a little bit more information - which bigrams did we miss?</summary>

1. We also know that `i` is likely to start a sequence.
2. And `a` is likely to end a sequence.

</details>

<details>
<summary>How can we include this new information in our training set so we don't miss it?</summary>

We can introduce special tokens - `<S>` for "start-of-sequence/word token", and `<E>` for "end-of-sequence/word token".

We would then get this:

1. `<S>i` (short for: `<S>` is followed by `i`)
2. `is` (short for: `i` is followed by `s`)
3. `sa` (short for: `s` is followed by `a`)
4. `ab` (short for: `a` is followed by `b`)
5. `be` (short for: `b` is followed by `e`)
6. `el` (short for: `e` is followed by `l`)
7. `ll` (short for: `l` is followed by `l`)
8. `la` (short for: `l` is followed by `a`)
9. `a<E>` (short for: `a` is followed by `<E>`)

</details>

<details>
<summary>How can we use this information to proceed with our task?</summary>

1. We can **count** the number of times each possible bigram appears in the training set.
2. We can use those counts to get a **probability** distribution over the next character given a current one.
3. We can then **sample** from this distribution (each character will get a weight) - this is also known as the [Multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution).

</details>

<details>
<summary>Technical question: how would it be best for us to store those counts?</summary>

Since the possible bigrams are not that much - `28` (including `SOS` and `EOS`), we can just have a matrix of size `28x28` in which each element:

- is of the form `xy`, meaning `the character "x" is followed by the character "y"`;
- holds how many times `the character "x" is followed by the character "y"` in the training dataset.

By doing so we'll get a matrix that looks like this:

![counts_01](assets/w15_counts_01.png?raw=true "counts_01.png")

</details>

<details>
<summary>What are two problems with this approach?</summary>

It stores unnecessary information - the bottom row is only zeros as is the second to last column. This is because:

- it's impossible for something to follow the `<E>` token;
- and it's impossible for the `<S>` token to follow a character.

Thus, we are wasting space enumerating impossible situations (bigrams).

</details>

<details>
<summary>How can we overcome this?</summary>

Let's replace `<S>` and `<E>` with `.` and place the enumerations of `.` to be the beginning of the matrix.

We would then get this:

![counts_02](assets/w15_counts_02.png?raw=true "counts_02.png")

</details>

<details>
<summary>What would our next step be and why?</summary>

We'd convert each row from counts to probabilities.

This is so that, for example, when we have the character `.` (beginning of a sentence) we can tell what's the most likely next character.

</details>

## The Multinomial Distribution

<details>
<summary>What do you know about it?</summary>

Ok, let's explain it with examples.

</details>

- You have a coin.
- Your experiment is throwing that coin $1$ time and checking whether it landed on `heads`. It's either going to be $1$ or $0$.

<details>
<summary>What distribution do you use to model these 0s and 1s?</summary>

[Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution): $K = 2$ outcomes, $n = 1$ trial.

</details>

- You have a coin.
- Your experiment is throwing that coin $n$ times and counting the total number of times it landed on `heads`. The result is a number from $0$ to $n$.

<details>
<summary>What distribution do you use to model this count?</summary>

[Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution): $K = 2$ outcomes, $n >= 1$ trial.

</details>

- You now have a dice.
- Your experiment is throwing that dice $1$ time and checking whether it landed on $6$. It's either going to be $1$ or $0$.

<details>
<summary>What distribution do you use to model these 0s and 1s?</summary>

[Categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution): $K >= 2$ outcomes, $n = 1$ trial.

</details>

- You have a dice.
- Your experiment is throwing that dice $n$ times and counting:
  - the total number of times it landed on $1$
  - and the total number of times it landed on $2$
  - and the total number of times it landed on $3$
  - and the total number of times it landed on $4$
  - and the total number of times it landed on $5$
  - and the total number of times it landed on $6$

<details>
<summary>What distribution do you use to model these counts?</summary>

[Multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution): $K >= 2$ outcomes, $n >= 1$ trial(s).

- Generalization of the binomial distribution.
- The number of outcomes is allowed to be greater than $2$.
- `You give me probabilities for integers and I'll give you integers that are sampled according to those probabilities.`
- Available in PyTorch as [`torch.multinomial`](https://pytorch.org/docs/stable/generated/torch.multinomial.html#torch-multinomial).

</details>

## Sampling from a multinomial distribution with PyTorch

### The [`Generator` class](https://pytorch.org/docs/stable/generated/torch.Generator.html#generator) in PyTorch

- If we regard `torch.manual_seed` as the "functional" way to set the seed, then using `Generator` is the "OOP" way.
- Used as a keyword argument in many [in-place random sampling functions](https://pytorch.org/docs/stable/torch.html#inplace-random-sampling).

### Sampling with a `Generator`

```python
g = torch.Generator().manual_seed(42)
p = torch.rand(3, generator=g)
p
```

The above would **always** produce:

```console
tensor([0.8823, 0.9150, 0.3829])
```

Let's normalize and draw from the multinomial distribution:

```python
import torch
g = torch.Generator().manual_seed(42)
p = torch.rand(3, generator=g)
p /= p.sum()
print(p) # tensor([0.4047, 0.4197, 0.1756])
drawn = torch.multinomial(p, num_samples=1000, replacement=True, generator=g)
print(f'Proportion of 0: {drawn[drawn == 0].shape[0] / drawn.shape[0]}') # Proportion of 0: 0.404
print(f'Proportion of 1: {drawn[drawn == 1].shape[0] / drawn.shape[0]}') # Proportion of 1: 0.425
print(f'Proportion of 2: {drawn[drawn == 2].shape[0] / drawn.shape[0]}') # Proportion of 2: 0.171
```

Notice how the probability of each index dictates how often it gets drawn.

<details>
<summary>So, if we have to use our model to create a new name, what are the steps to follow?</summary>

1. Create a string with a `.`.
2. Get the row that stores the probabilities of the characters that can follow `.`.
3. Sample an index using the probabilities.
4. Look up that index to see which letter it corresponds to. Let's say it's `m`.
5. Add `m` to the initial string. It is now `.m`.
6. Get the row that stores the probabilities of the characters that can follow `m`.
7. Sample an index using the probabilities.
8. Look up that index to see which letter it corresponds to. Let's say it's `a`.
9. Add `a` to the initial string. It is now `.ma`.
10. And so on...

</details>

<details>
<summary>When do we end the process?</summary>

Two possibilities:

- either we sample the character `.` again;
- or we have a predefined `max_sequence_length` and we stop once that many characters are sampled.

</details>

## Negative Log Likelihood

Let's explore how we can evaluate the quality of our model.

<details>
<summary>How would the baseline model work?</summary>

The baseline / "dumbest" model would predict everything as equally likely, so each probability of a bigram pair would be $1/27 = 0.037$.

</details>

<details>
<summary>How would we know that our model has learned anything?</summary>

If our model assigns a probability above $0.037$ to at least one pair, then it has learned some pattern.

</details>

Let's say that after we create the table with the probabilities, we check what probabilities the model assigns to the bigrams formed by the first `3` names and we get this:

![w15_probs_tr_set.png](assets/w15_probs_tr_set.png "w15_probs_tr_set.png")

<details>
<summary>In the ideal situation what values would we want to have?</summary>

We would want to have all values close to $1$, since that means that the model has learned the pattern in the training set and thus can recreate the training set (which has real names).

</details>

<details>
<summary>Can we calculate a single number that would say how likely the model is to generate the names it was trained on?</summary>

We can estimate the likelihood of the data (i.e. that the model would generate names like the ones it was trained on). This would be equivalent to the **product of all probabilities** assigned by the model for all training bigrams.

</details>

<details>
<summary>What would a good model do with this number?</summary>

It would [maximize the likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).

Let's define it mathematically:

${\displaystyle L = \prod_{i=1}^n p(y_i)}$

</details>

<details>
<summary>What is the problem of directly applying this?</summary>

- All of the probabilities are between $0$ and $1$.
- If we multiply them we're going to get a very small number:
  - It will become even smaller as more probabilities are present.

</details>

<details>
<summary>How can we solve this problem?</summary>

We can take the $\log$ of the product!

$\log$ is:

- monotonically increasing;
- can better "distinguish" between very small numbers (we plug-in $x$, it gives us $y$);

![w15_log_from_0_to_1.png](assets/w15_log_from_0_to_1.png "w15_log_from_0_to_1.png")

- and has this very nice equivalence: $\log (abc) = \log a + \log b + \log c$
- the base is usually $e$, though it can vary.

The formula would then become:

${\displaystyle L = \sum_{i=1}^n \ln(p(y_i))}$

</details>

<details>
<summary>How large can that value get?</summary>

The largest possible log-likelihood is $0$.

</details>

<details>
<summary>Why is this a problem?</summary>

Because these are not the semantics we want out of a loss function. We want to minimize loss functions.

</details>

<details>
<summary>How can we solve this?</summary>

We can measure the negative log-likelihood!

The formula would then become:

${\displaystyle L = - \sum_{i=1}^n \ln(p(y_i))}$

</details>

And finally, for convenience, we usually normalize the values of the loss functions using the number of used samples, so the final version of the $NLL$ loss is:

${\displaystyle NLL = -\frac{1}{n} \sum_{i=1}^n \ln(p(y_i))}$

We've heard about the loss function [Cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy). It's actually the same as calling $NLL$ after passing the logits through $Softmax$:

- Here's the implementation of it in C++: <https://github.com/pytorch/pytorch/blob/e0ff6a21d8bf3935f4332fb18b40b48472f9b648/aten/src/ATen/native/LossNLL.cpp#L607-L612>.
- So, then:

```python
F.nll_loss(-log(softmax(logits, dim=1)),targets)
```

is equivalent to:

```python
F.cross_entropy(logits,targets)
```

But wait - this:

$$\text{NLL} = -\frac{1}{n} \sum_{i=1}^n \ln(p(y_i))$$

is different from this

$$\text{CrossEntropy}(y, \hat{y}) = -\frac{1}{n} \sum_{j=1}^{n} \sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

, right?

Actually, they are the same thing.

- $y_i$ is the true label for class $i$. In most classification tasks, this is one-hot encoded, meaning:
  - $y_i = 1$ for the correct class $t$;
  - $y_i = 0$ for all other classes.

For any single sample only one $y_i$ will be $1$, so we can simplify the summation to:

$$\text{CrossEntropy}(y, \hat{y}) = -\log(\hat{y}_t)$$

Let's show this via an example. Suppose we have $3$ classes, and the true class is class $2$. The model outputs:

$$\hat{y} = [0.1, 0.7, 0.2]$$

Then the true label vector is:

$$y = [0, 1, 0]$$

The cross entropy loss becomes:

$$-\left(0 \cdot \log(0.1) + 1 \cdot \log(0.7) + 0 \cdot \log(0.2)\right) = -\log(0.7)$$

## Model Smoothing and Regularization

In the previous section we saw how we can calculate the probability that our model would predict any given name. We start testing it out and for the word `jaguar` the model returns $-3.5$.

<details>
<summary>Does this mean that the model is likely to generate it or no?</summary>

This is far from $0$, so it's not very likely to generate it.

</details>

<details>
<summary>What number would we expect to get for the word "jqaguar" compared to "jaguar"?</summary>

We would expect something more negative, but actually if we run this through the model, we would get $\inf$!

</details>

<details>
<summary>Why is that?</summary>

By storing the true counts we start to miss edge-cases (the model loses its novelty):

- it won't be able to generate bigrams it hasn't seen:
  - the counts would be `0` as the training set does not contain examples of them;
  - thus, the model would assign `probability=0` if checked for unseen bigrams.
  - when the loss is calculated, we can get $\inf$ because `log(0)` is `undefined`.

But this is not ok - the model should be able to generate a name that is rather uncommon, but still possible - uncommon names exist in real life.
</details>

<details>
<summary>How can we fix this?</summary>

We can utilize a technique called **model smoothing** to overcome this:

- smoothing is used to regulate the probability assigned to words that have unseen bigrams;
- it has various forms: you can read more [here](http://mlwiki.org/index.php/Smoothing_for_Language_Models) and [here](https://github.com/akash18tripathi/Language-Modelling-and-Smoothing-techniques?tab=readme-ov-file#smoothing-in-language-modeling);
- we'll use a very simple form: just start counting from $1$ instead of $0$;
  - we can add smoothing at the step in which we're turning the counts into probabilities - instead of turning the raw counts, we'll use `counts + 1`. This variation is known as `+1 smoothing`.

Using `+1 smoothing` ensures every possible bigram, even those not present in our training dataset, has been "seen" by the model at least `1` time. As expected very low probabilities would be assigned to words containing unseen bigrams, but at least they wouldn't be zero.

</details>

<details>
<summary>What would happen if we start increasing those "fake" counts - what if it was 1000 instead of 1?</summary>

This would lead to more uniformly distributed probabilities all bigrams will be more equally likely to get predicted.

</details>

<details>
<summary>So in the end it seems that this is making our model dumber in exchange to produce unconventional results. What is this technique (that introduces more bias to reduce the variance) called?</summary>

Model regularization!

</details>
