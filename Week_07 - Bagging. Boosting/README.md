# Goals for week 07

1. Practice working the technique of bagging.
2. Practice working the technique of boosting.
3. Implement a Random Forest classifier.
4. Implement `Adaboost`.

## Data Science

Learning how to model data effectively.

### Task 01

**Description:**

Last week we created a model that predicts whether a patient has liver disease. Try to improve the achieved performance by trying out bagging and boosting classifiers, including gradient boosting techniques, XGBoost and CatBoost (last two have their own dedicated Python packages).

Set the baseline model to last week's best model and try to improve it. Conduct hyperparameter tuning.

For the bagging algorithms, besides the usual metric(s), report the out-of-bag metric as well (don't use the default behavior).

Apart from the ROC curve attach a Precision-Recall curve and a horizontal bar plot from the feature importances. Interpret the results.

**The goal of this task is to explore bagging and boosting algorithms and their hyperparameters. Do not shy away from many experiments with all of the available hyperparameters!**

**Acceptance criteria:**

1. An Excel file showing the model report is produced.
2. The baseline is improved by at least $5%$.

### Task 02

**Description:**

Predict bike rental demand in the Capital Bikeshare program in Washington, D.C, using historical weather data from the dataset `bike_sharing.csv` in our `DATA` folder.

Here is a description of the fields:

- `datetime`: hourly date + timestamp
- `season`:  1 = spring, 2 = summer, 3 = fall, 4 = winter
- `holiday`: whether the day is considered a holiday
- `workingday`: whether the day is neither a weekend nor holiday
- `weather`:
  - `1`: Clear, Few clouds, Partly cloudy, Partly cloudy
  - `2`: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
  - `3`: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
  - `4`: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- `temp`: temperature in Celsius
- `atemp`: "feels like" temperature in Celsius
- `humidity`: relative humidity
- `windspeed`: wind speed
- `casual`: number of non-registered user rentals initiated
- `registered`: number of registered user rentals initiated
- `count`: number of total of bike rentals. This is the target feature.

Compare all available regression algorithms you know, including bagging and boosting regressors, including gradient boosting techniques, XGBoost and CatBoost.

Apart from the graph showing how the residuals vary around $0$ for the best model, attach a horizontal bar plot with the feature importances.

**The goal of this task is to explore bagging and boosting algorithms and their hyperparameters. Do not shy away from many experiments with all of the available hyperparameters!**

**Acceptance criteria:**

1. An Excel file showing the model report is produced.

## Engineering

Building systems and implementing models.

### Task 01

**Description:**

Add a random forest classifier to `ml_lib.tree`. Implement all functionalities you deem relevant that are currently not in the package. The class should have at least the methods `fit`, `predict` and `score`.

It is up to your decision on what hyperparameters you'd implement.

**Acceptance criteria:**

1. A class `RandomForestClassifier` is added to `ml_lib.tree`.

### Task 02

**Description:**

Use your implementation of the `RandomForestClassifier` to reproduce the metrics of the best random forest classifier obtained in `Task 01` in the section `Data Science`.

**Acceptance criteria:**

1. The best `RandomForestClassifier` is recreated using `ml_lib`.

### Task 03

**Description:**

Add an `AdaBoostClassifier` classifier to `ml_lib.tree`. Implement all functionalities you deem relevant that are currently not in the package. The class should have at least the methods `fit`, `predict` and `score`.

It is up to your decision on what hyperparameters you'd implement.

**Acceptance criteria:**

1. A class `AdaBoostClassifier` is added to `ml_lib.tree`.

### Task 04

**Description:**

Use your implementation of the `AdaBoostClassifier` to reproduce the metrics of the best random forest classifier obtained in `Task 01` in the section `Data Science`.

**Acceptance criteria:**

1. The best `AdaBoostClassifier` is recreated using `ml_lib`.
