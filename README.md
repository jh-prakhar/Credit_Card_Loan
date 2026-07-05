# Credit Card Loan Approval Analysis

An end-to-end Machine Learning minor project designed to automate and predict the credit card loan approval process based on applicant profiles. This project leverages historical financial records and demographic factors to classify whether an applicant will be approved or rejected.

## 📌 Project Overview
Manual verification of credit card and loan eligibility is time-consuming and prone to human error. This project uses historical applicant data to train predictive classification algorithms, minimizing financial default risks while accelerating the approval workflow.

---

## 🛠️ Project Workflow

### 1. Data Ingestion
* The structured customer demographic and financial dataset is imported into the Python environment using the `pandas` library.

### 2. Data Cleaning & Preprocessing
* **Missing Value Treatment**: Handled null and missing observations across columns using appropriate statistical imputation techniques to prevent algorithmic bias.

### 3. Exploratory Data Analysis (EDA)
Comprehensive visual distributions were generated using `matplotlib` and `seaborn` to understand feature impacts:
* **Loan Approval Ratio**: Analyzed the general distribution of target variables using a **Pie Chart**.
* **Category Analysis**: Examined categorical metrics across distinct clusters using a **Barplot**.
* **Income Distribution**: Identified applicant earning patterns using a **Histplot**.
* **Outlier Detection**: Discovered anomalies and severe variances in continuous variables using a **Boxplot**.
* **Credit Score Impact**: Explored the deep relationship between Credit Score and final Loan Approval status using an overlapping **Histplot**.

### 4. Feature Engineering & Categorical Encoding
To prepare text and category columns for machine learning algorithms, two distinct encoding approaches were executed:
* **Label Encoding**: Applied to ordinal structures where hierarchy matters:
  * `Education level`
  * `Loan Approved` (Target variable)
* **One-Hot Encoding**: Applied to non-ordinal categorical values to prevent false numerical ordering:
  * `Employment Status`, `Marital Status`, `Loan Purpose`, `Property Area`, `Gender`, and `Employment Category`.

### 5. Correlation Analysis
* Constructed a **Correlation Heatmap** to assess linear dependencies among metrics, identifying multicollinearity and highlighting primary features strongly tied to loan issuance.

### 6. Initial Data Splitting
* Utilized the standard **Train-Test Split** validation mechanism.
* **80% of the dataset** was allocated to train and build the models.
* **20% of the data** was strictly isolated for testing and final evaluation performance.

### 7. Machine Learning Modeling (Phase 1)
The preprocessed training matrix was fitted into classification models:
* **Logistic Regression**: Serves as the baseline linear model to calculate probability boundaries for credit approval.
* **k-Nearest Neighbors (kNN) Model**: Applied as a distance-based, non-parametric approach to classify applicant eligibility based on similar profile proximity.
* **Naive Bayes**: Utilized as a probabilistic classifier based on Bayes' Theorem.

### 8. Advanced Feature Engineering
To capture non-linear relationships, reduce skewness, and optimize predictive power, specialized data transformations were implemented:
* **Polynomial Transform (Square)**: Squared the values of `Debt to Income Ratio` and `Credit Score` to penalize risky profiles heavily.
* **Log Transformation (Log1p)**: Applied a natural log transformation `log(x + 1)` to `Applicant Income` to normalize its heavily right-skewed distribution and minimize the impact of extreme outliers.

### 9. Feature Scaling & Final Split
* Re-split the newly transformed data matrix using the exact **80% training and 20% testing split**.
* **Feature Scaling**: Implemented a scaling normalization method to ensure magnitude equality across distance-sensitive features.

---

## 🏆 Model Evaluation & Performance Results

Following feature engineering, transformations, and scaling, all three models were tested on the remaining 20% of the data. 

### Performance Comparison Matrix

| Machine Learning Model | Accuracy Score | Precision Score | Recall Score | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| 🥇 **Naive Bayes** | 86.50% | **80.36%** | 73.77% | 76.92% |
| 🥈 **Logistic Regression** | **87.00%** | 77.78% | **80.33%** | **79.03%** |
| 🥉 **k-Nearest Neighbors (kNN)** | 77.00% | 68.29% | 45.90% | 54.90% |

### Confusion Matrix Insights

#### 1. Naive Bayes (Best for Risk Management)
* **True Negative (TN):** 128 | **False Positive (FP):** 11
* **False Negative (FN):** 16 | **True Positive (TP):** 45
* *Analysis:* While Logistic Regression achieved slightly higher baseline accuracy, **Naive Bayes is the best choice for this project**. It yielded the highest **Precision Score (80.36%)** and generated the lowest number of False Positives (only 11). In banking, reducing False Positives is vital because it stops the system from approving bad, high-risk loans.

#### 2. Logistic Regression
* **True Negative (TN):** 125 | **False Positive (FP):** 14
* **False Negative (FN):** 12 | **True Positive (TP):** 49
* *Analysis:* Solid overall performer with the highest raw accuracy (87.00%) and excellent sensitivity (80.33% Recall) for catching eligible candidates.

#### 3. k-Nearest Neighbors (kNN)
* **True Negative (TN):** 126 | **False Positive (FP):** 13
* **False Negative (FN):** 33 | **True Positive (TP):** 28
* *Analysis:* Struggled significantly with high-dimensional scaled spaces, resulting in a low Recall score (45.90%) and a high rate of missed approvals (33 False Negatives).

---

## 💻 Tech Stack & Libraries Used
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `seaborn`, `matplotlib`
* **Machine Learning Framework:** `scikit-learn`

---

## 🚀 How to Run This Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd credit-card-loan-analysis
   ```

2. **Install Required Packages:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Execute the Pipeline:**
   Open your development environment and run your main script file or Jupyter Notebook to execute the pipeline cell by cell.
