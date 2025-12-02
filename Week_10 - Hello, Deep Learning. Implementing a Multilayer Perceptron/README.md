# Goals for week 10

1. Create a neural network that can multiply a number by `2`.
2. Create neural networks that simulate the logical gates `AND`, `OR`, `NAND` and `XOR`.

## Task 1

**Description:**

We want to create and train a deep learning model that given `x` predicts `2 * x`.

Define the following functions:

1. `create_dataset`: accepts `n` and returns `n` consecutive samples that demonstrate the expected behavior.
2. `initialize_weights`: accepts `x` and `y` and returns a random number from a uniform distribution in the range `[x, y)`.

**Acceptance criteria:**

1. A general form of the model is placed in a comment that shows how many parameters the model has.
2. `numpy` is used to initialize the parameter(s) of the model.

**Test case:**

Due to randomness your results may vary.

```python
create_dataset(4)          # [(0, 0), (1, 2), (2, 4), (3, 6)]
initialize_weights(0, 100) # 95.07143064099162
initialize_weights(0, 10)  # 3.745401188473625
```

## Task 2

**Description:**

We want to create and train a machine learning model that given `x` predicts `2 * x`.

Create a dataset with `6` samples.
Create a model that has a random value for its parameter in the interval `[0, 10)` with a seed set to `42`.

Define a function `calculate_loss` that accepts the model, a dataset and computes the mean squared error of the model over the dataset.

Experiment with the first parameter of the model - what happens to loss function when you pass `w + 0.001 * 2`, `w + 0.001`, `w - 0.001` and `w - 0.001 * 2`?

**Acceptance criteria:**

1. The above functionality is implemented.
2. A comment is present with an answer to the question in the description.

**Test case:**

Due to randomness your result may vary.

```python
loss = calculate_loss(w, dataset)
print(f'MSE: {loss}') # MSE: 27.92556532998047
```

## Task 3

**Description:**

We want to automatically update the value of the parameter using the method of finite differences to approximate the derivative. Approximate the derivative and subtract the resulting value from the parameter. Investigate how adding a `learning rate` of `0.001` impacts the value of the loss function.

**Acceptance criteria:**

1. The above functionality is implemented.
2. Print the value of the loss function before updating the parameter.
3. Print the value of the loss function after updating the parameter.
4. Experiments are done that show what impact adding a learning rate has on updating the parameter(s).
5. Include a `for-loop` in which the process is repeated `10` times (also known as `epochs` - the number of times the whole dataset is traversed).

## Task 4

**Description:**

Train the model for `500 epochs` and print the value of parameter(s).

Experiment with removing the `seed` and seeing whether the training still converges.

Congratulations! You just created and trained your first neural network 🎉!

**Acceptance criteria:**

1. The above functionality is implemented.

## Task 5

**Description:**

The above tasks result in a neural network with one neuron and one input. In this task we're going to model the `AND` and `OR` circuit operations, thus the network is going to have two inputs and thus two weights attached to them.

Create two models and train for `100,000` epochs and after each epoch print the values for the parameters of the models and the value of the loss function. When the two models are trained, apply each of them to their corresponding training sets and print the values they predict.

Print the values the models predict. What do you notice about the confidence the model has in them?

**Acceptance criteria:**

1. A general form of the two models is placed in a comment that shows how many parameters they have.
2. A dataset of tuples with three elements is created for the `OR` gate.
3. A dataset of tuples with three elements is created for the `AND` gate.
4. The loss function is modified accordingly.
5. Weights are initialized randomly without seed.
6. The loss function's output values are smaller as more epochs are done.
7. The two models are trained for `100,000` epochs.
8. A comment is present with an answer to the question in the description.

## Task 6

**Description:**

Extend the functionality of the previous task by adding a free-floating parameter (i.e. a parameter that has a weight of `1`) - the so-called `bias`. This would help the model drive the loss down to `0`. Our models will now have three parameters, though, note that the `bias` is a parameter of the model itself, it's not part of the dataset.

The `bias` parameter allows the model to shift its prediction when all of its weights become `0`.

Print the values the models predict. What has changed in comparison to task 5?

**Acceptance criteria:**

1. A comment is present with an answer to the question in the description.

## Task 7

**Description:**

Add a sigmoid function and plot the values it takes from `-10` till `10` using `matplotlib`.

**Acceptance criteria:**

1. The above functionality is implemented.
2. A plot is produced using `matplotlib`.

## Task 8

**Description:**

Use the sigmoid function during model training and inference. What happens to the value of the loss function?

Using `matplotlib` plot how the loss changes for `100,000` epochs. Plot the epoch on the `x`-axis and the loss on the `y`-axis.

**Acceptance criteria:**

1. A comparison is made before and after applying sigmoid in a comment.

## Task 9

**Description:**

Create a dataset and a model for the `NAND` logic gate. Can you reuse the already created models for the `AND` and `OR` gates and just change their datasets?

**Acceptance criteria:**

1. The above functionality is implemented.

## Task 10

**Description:**

Create a Python class named `Xor` and store all parameters and biases of a `Xor` model that implements the `XOR` logic gate. Train and evaluate the model using an appropriate architecture.

**Acceptance criteria:**

1. High quality code is written:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.
2. A `forward` function is implemented that takes a model, two inputs and returns the output of the model.

## Task 11

**Description:**

Answer the following questions in Bulgarian:

1. What is machine learning in 1 sentence?
2. What is deep learning in 1 sentence?
3. What is does model training mean in 1 sentence?
4. What task do Large Language Models solve?
5. What is the difference between loss functions and metrics?
6. Why is the use of mean squared error preferred to the use of mean absolute error?
7. Why can't we cube the difference to get the mean error instead of squaring it?
8. Why is the derivative of the loss function beneficial during model training?

**Acceptance criteria:**

1. Answers are written in Bulgarian.
2. Answers are written in the body of the email (not in a `.docx` or `.txt` file).

## Task 12

**Description:**

Watch [this](https://www.youtube.com/watch?v=2kSl0xkq2lM) YouTube video and answer the following questions:

1. Which class of AI techniques began to work around 2005 for solving real-world problems?
2. What is a classic application of AI is discussed in the video?
3. What is the simplest way of getting Machine Learning to solve a task?
4. Define what supervised learning is.
5. What does AI require in order to work at all?
6. What class of tasks does the task for facial recognition belong to?
7. What is the task solved by each neuron in the brain?
8. What happens when a neuron recognizes a pattern?
9. What is a digital picture made up of?
10. What are the three reasons that allowed the modelling of human brain cells in software?
11. Which type of computer processor is suited really well for performing the mathematics related to deep learning?
12. Do the capabilities of neural networks grow with scale?
13. What allowed for the biggest and fastest advances in the field?
14. What is the name of the probably most important paper in the last decade?
15. What is the Transformer architecture designed for?
16. What is the innovation that the Transformer architecture has?
17. What is the key point about GPT-3?
18. Is machine learning more efficient at learning than humans? Why?
19. What is "the bitter truth"?
20. What is one reason why ChatGPT is not conscious?

**Acceptance criteria:**

1. Answers are written in Bulgarian.
2. Answers are written in the body of the email (not in a `.docx` or `.txt` file).
