# Goals for week 14

1. Practice working with recurrent neural networks.

## Task 01

**Description:**

Let's tackle the problem of predicting electricity consumption based on past patterns. In our `DATA` folder you'll find the folder `electricity_consumption`. It contains electricity consumption in kilowatts, or kW, for a certain user recorded every 15 minutes for four years.

First, define a function `create_sequences` that takes a `pandas` `Dataframe` and a sequence length and returns two `NumPy` arrays, one with input sequences and the other one with the corresponding targets. To test the correctness of the function, apply it on a dataset containing two columns with the numbers from `0` to `100` and output the first five entries in both arrays.

Second, apply the function on the training set of the electricity consumption data. Output the shape of the `X` and `y` splits and output the length of the `TensorDataset` for training the model.

**Acceptance criteria:**

1. A new function `create_sequences` is defined.
2. The first five entries after applying `create_sequences` on the `electricity_consumption` data are displayed.
3. An object of type `TensorDataset` is created and its length is outputted.

**Test case:**

```console
python task01.py
```

```console
Validating "create_sequences"
First five training examples in dataset with the numbers from 0 to 100: [[0 1 2 3 4]
 [1 2 3 4 5]
 [2 3 4 5 6]
 [3 4 5 6 7]
 [4 5 6 7 8]]
First five target values: [5 6 7 8 9]

Applying "create_sequences" on the electricity consumption data
X_train.shape=(105119, 96)
y_train.shape=(105119,)
Length of training TensorDataset: 105,119
```

## Task 02

**Description:**

Create a recurrent neural network that predicts electricity consumption amounts.

Use the `tqdm` library to visualize the training process. Output the loss per epoch when training and the loss on the test set.

**Acceptance criteria:**

1. An Excel file, titled `model_report`, is created which follows **all** guidelines.
2. `tqdm` is used to visualize the progress through the batches.
