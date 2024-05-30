# Predicting Stars, Galaxies & Quasars with ML Model

This project focuses on building machine learning models to classify celestial objects into stars, galaxies, and quasars using a provided dataset. The following sections detail the workflow and the steps taken in the notebook.

## Workflow of the Notebook

1. **Introducing Dataset**
2. **Importing Necessary Libraries and Modules**
3. **Exploring the Dataset**
4. **Preparing Data for the Model**
5. **Scaling the Data and Checking Distribution Plots**
6. **Building ML Models and Evaluating Results**

## Dataset Introduction

The dataset used for this project contains information about celestial objects. The classification task involves predicting whether an object is a star, galaxy, or quasar based on various features provided in the dataset.

## Importing Necessary Libraries and Modules

We utilize several Python libraries for data manipulation, visualization, and building machine learning models. Here is the list of the libraries:

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

%matplotlib inline
```

## Loading and Exploring the Dataset

### Importing the Dataset

```python
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data = pd.read_csv("/content/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
data.head()
```

### Dataset Dimensions and Preliminary Analysis

```python
data.shape
data.describe()
```

### Dropping Unnecessary Columns

```python
data.drop(['objid','specobjid'], axis=1, inplace=True)
data.head(10)
```

### Checking for Null Values

```python
data.info()
```

The dataset is complete with no missing values.

### Encoding Target Variable

```python
le = LabelEncoder().fit(data['class'])
data['class'] = le.transform(data['class'])
data.head(10)
```

### Final Dataset Information

```python
data.info()
```

## Data Preparation

### Splitting Data into Features and Target

```python
X = data.drop('class', axis=1)
y = data['class']
```

## Data Scaling

Standardizing the dataset to have a mean of 0 and a standard deviation of 1.

```python
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X = scaler.fit_transform(X)
```

## Building Machine Learning Models

We employ several machine learning models to classify the data:

- **Decision Tree Classifier**
- **Logistic Regression**
- **Naive Bayes**
- **K-Nearest Neighbors**
- **Support Vector Machine**

### Training and Evaluating the Models

```python
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example for Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred)}")

# Repeat similar steps for other models
```
## license
This program is under the [MIT License](LICENSE)

## Conclusion

This notebook demonstrates the process of loading a dataset, performing exploratory data analysis, preparing the data, scaling it, and finally building and evaluating several machine learning models to classify celestial objects.

## References

- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)

---

Upload this notebook to a Kaggle session, link the dataset, and run the cells to reproduce the analysis and results.
