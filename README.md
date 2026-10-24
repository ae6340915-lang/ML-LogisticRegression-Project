# 🍔 Online Food Delivery — Machine Learning Classification

A Machine Learning classification project using **Logistic Regression** to predict whether a customer will place an online food delivery order (`Yes` or `No`) based on customer demographics, income, occupation, family size, location, and feedback.

This project demonstrates a complete Machine Learning workflow, including data cleaning, exploratory data analysis, categorical encoding, feature scaling, model evaluation, ROC analysis, and sequential feature selection.

## 📌 Project Overview

The goal of this project is to build and evaluate a binary classification model using customer data from an online food delivery dataset.

The target variable is `Output`:

* `Yes` → Customer orders food online.
* `No` → Customer does not order food online.

The project also investigates feature relationships and uses Sequential Feature Selection to identify a smaller subset of potentially useful features.

## 🎯 Project Objectives

* Explore and understand customer data.
* Inspect data types, statistical summaries, missing values, and duplicate records.
* Clean the dataset and remove irrelevant columns.
* Perform exploratory data analysis (EDA).
* Encode categorical variables using one-hot encoding.
* Analyze correlations between features.
* Standardize features using `StandardScaler`.
* Train a Logistic Regression classifier.
* Evaluate classification performance using multiple metrics.
* Analyze the ROC curve and ROC-AUC score.
* Apply Forward Sequential Feature Selection to select eight features.

## 🛠️ Technologies & Libraries

| Technology       | Purpose                                                          |
| ---------------- | ---------------------------------------------------------------- |
| Python           | Main programming language                                        |
| Pandas           | Data manipulation and cleaning                                   |
| NumPy            | Numerical operations                                             |
| Matplotlib       | Data visualization                                               |
| Seaborn          | Statistical visualization and heatmaps                           |
| Scikit-learn     | Preprocessing, classification, feature selection, and evaluation |
| Jupyter Notebook | Interactive development                                          |

## 📂 Dataset

The dataset contains customer information related to online food delivery.

**Original dataset dimensions:** 388 rows × 14 columns

**After cleaning:** 285 rows × 12 columns

The dataset includes the following types of information:

* **Demographics:** Age, Gender, Marital Status
* **Employment and income:** Occupation, Monthly Income
* **Education:** Educational Qualifications
* **Household:** Family size
* **Customer behavior:** Customer Type, Feedback
* **Geographical information:** Latitude, Longitude
* **Target variable:** Output

Two columns, `Unnamed: 13` and `Pin code`, were removed during preprocessing. The remaining duplicate records were also removed.

After cleaning, the dataset contained 285 records.

## 🔄 Machine Learning Workflow

### 1. Data Loading and Inspection

Loaded the CSV dataset using Pandas and examined its structure with:

* `head()`
* `info()`
* `columns`
* `describe()`
* `isnull().sum()`

The original dataset contained 388 rows and 14 columns, with no missing values reported.

### 2. Data Cleaning

Performed the following operations:

* Removed the irrelevant `Unnamed: 13` column.
* Removed `Pin code` from the feature set.
* Trimmed whitespace from categorical values.
* Checked for missing values and duplicate records.
* Removed duplicate rows.

**Cleaning results:**

| Stage                      | Rows | Columns |
| -------------------------- | ---: | ------: |
| Original dataset           |  388 |      14 |
| After removing two columns |  388 |      12 |
| After removing duplicates  |  285 |      12 |

The dataset contained 103 duplicate rows before duplicate removal.

### 3. Exploratory Data Analysis (EDA)

Examined the dataset's distributions and characteristics using descriptive statistics and visualizations.

The analysis included:

* Customer age and family-size statistics.
* Target-class distribution.
* Feedback distribution.
* Feature correlation analysis.

Before duplicate removal, the target variable contained:

| Target Class | Records |
| ------------ | ------: |
| Yes          |     301 |
| No           |      87 |

The feedback distribution before duplicate removal was:

| Feedback | Records |
| -------- | ------: |
| Positive |     317 |
| Negative |      71 |

These counts describe the original dataset before cleaning and should not be interpreted as the final class distribution.

### 4. Target Preparation and Train-Test Split

Separated the target variable from the input features.

* `X`: Input features.
* `y`: Binary target derived from `Output`.
* `Yes` was encoded as `1`.
* `No` was encoded as `0`.

Used a stratified train-test split to preserve the target-class proportions.

| Dataset      | Records |
| ------------ | ------: |
| Training set |     228 |
| Testing set  |      57 |
| Total        |     285 |

Configuration:

* Test size: 20%
* Random state: 42
* Stratification: Enabled

### 5. Categorical Encoding

Converted categorical variables into numerical features using:

`pd.get_dummies(drop_first=True, dtype=int)`

Aligned the training and testing columns to ensure both datasets had matching feature structures.

| Dataset                     | Rows | Features |
| --------------------------- | ---: | -------: |
| Training set after encoding |  228 |       21 |
| Testing set after encoding  |   57 |       21 |

### 6. Correlation Analysis

Calculated a correlation matrix for the encoded training features.

Created a Seaborn heatmap to highlight correlations where the absolute correlation coefficient was at least 0.5.

This visualization helps identify relationships between features and potential multicollinearity.

### 7. Feature Scaling

Applied `StandardScaler` to standardize the encoded training features and transform the testing features.

The scaler was fitted on the training data and then applied to the testing data.

**Important:** The scaled datasets were used for feature selection in the provided notebook. The Logistic Regression model evaluated below was trained on the unscaled encoded training data.

### 8. Logistic Regression Model

Trained a binary Logistic Regression classifier using Scikit-learn.

Model configuration:

* Algorithm: Logistic Regression
* Maximum iterations: 1,000
* Task: Binary classification

The model predicts whether a customer belongs to the `Yes` or `No` class.

### 9. Model Evaluation

Evaluated the model on the held-out testing set using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report
* ROC Curve
* ROC-AUC

### 10. ROC Curve and AUC

Calculated predicted probabilities using `predict_proba()` and evaluated the model's ability to distinguish between the two classes.

The ROC curve visualizes the relationship between the true positive rate and the false positive rate at different classification thresholds.

### 11. Forward Sequential Feature Selection

Applied Scikit-learn's `SequentialFeatureSelector` with:

* Estimator: Logistic Regression
* Selection direction: Forward
* Number of selected features: 8

The selected features were:

1. `Age`
2. `Family size`
3. `latitude`
4. `Marital Status_Prefer not to say`
5. `Marital Status_Single`
6. `Occupation_Student`
7. `Monthly Income_Below Rs.10000`
8. `Feedback_Positive`

This reduced the feature set from 21 features to 8.

**Note:** Feature selection was performed after evaluating the initial model. The selected features have not yet been used to train and evaluate a separate final model in the results reported here.

## 📈 Model Performance

The following results were obtained from the Logistic Regression model on the 57-record test set.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 82.46% |
| Precision | 85.11% |
| Recall    | 93.02% |
| F1-score  | 88.89% |
| ROC-AUC   | 0.7990 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
| ----- | --------: | -----: | -------: | ------: |
| No    |      0.70 |   0.50 |     0.58 |      14 |
| Yes   |      0.85 |   0.93 |     0.89 |      43 |

### Results Interpretation

* **Accuracy (82.46%):** The model correctly classified approximately 82% of the test records.
* **Precision (85.11%):** Of the records predicted as `Yes`, approximately 85% actually belonged to the `Yes` class.
* **Recall (93.02%):** The model identified approximately 93% of the actual `Yes` cases.
* **F1-score (88.89%):** The model achieved a strong balance between precision and recall for the `Yes` class.
* **ROC-AUC (0.7990):** The model demonstrated useful discriminatory ability between the two target classes.

The model performed better at identifying `Yes` cases than `No` cases. The recall for the `No` class was 50%, indicating that further evaluation and improvement are needed if correctly identifying non-ordering customers is important.

These results are based on a relatively small test set, so additional validation is recommended before drawing strong conclusions about generalization.

## ⚠️ Limitations and Future Improvements

Potential next steps include:

* Train a second Logistic Regression model using the eight selected features.
* Compare the full-feature model with the feature-selected model.
* Apply cross-validation for a more reliable performance estimate.
* Investigate class imbalance and evaluate suitable strategies.
* Compare Logistic Regression with other classification algorithms.
* Review feature importance and model coefficients.
* Use a preprocessing pipeline to make training and inference more consistent.
* Investigate whether `Feedback` is available at the time the prediction would actually be made, to avoid target leakage.
* Evaluate performance on additional unseen data.

## 📁 Project Structure

```text
ML-LogisticRegression-Project/
│
├── 4.ipynb
├── online food delivery dataset.csv
└── README.md
```

The dataset file should only be included in the public repository if redistribution is permitted.

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/ae6340915-lang/ML-LogisticRegression-Project.git
```

### 2. Navigate to the Project Folder

```bash
cd ML-LogisticRegression-Project
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Open the Notebook

```bash
jupyter notebook 4.ipynb
```

Alternatively, open the notebook using Visual Studio Code with the Jupyter extension installed.

### 5. Update the Dataset Path

The notebook currently uses a local Windows path:

```python
df = pd.read_csv(r"C:\pandas\online food delivery dataset.csv")
```

If you clone the repository on another computer, update the path to match your local dataset location. If the CSV file is stored in the repository root, you can use:

```python
df = pd.read_csv("online food delivery dataset.csv")
```

Run the notebook cells in order.

## 👨‍💻 Author

**GitHub:** [ae6340915-lang](https://github.com/ae6340915-lang)

**Project:** Online Food Delivery — Machine Learning Classification

---

⭐ This project demonstrates practical experience with data preprocessing, exploratory data analysis, binary classification, model evaluation, and feature selection.
