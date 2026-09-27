# 📊 Customer Churn Prediction — End-to-End Machine Learning Project

An end-to-end machine learning classification project for predicting customer churn using **Random Forest**, **SMOTE**, **hyperparameter optimization**, and model explainability techniques.

The project covers the complete machine learning workflow, from data preprocessing and exploratory data analysis to model optimization, interpretation, and deployment using **Streamlit**.

## 🎯 Project Objective

Customer churn is an important business problem because retaining existing customers can be more valuable than continuously acquiring new ones.

The objective of this project is to build a machine learning model that predicts whether a customer is likely to **churn** based on customer characteristics, services, contract information, and billing behavior.

The final model can be used through an interactive **Streamlit web application** where customer information is entered and the model returns:

- Churn prediction
- Churn probability

## 🔄 Project Workflow

1. Data Understanding
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Categorical Feature Encoding
5. Feature Scaling
6. Train/Test Split
7. Baseline Random Forest Model
8. Handling Class Imbalance with SMOTE
9. Hyperparameter Optimization with RandomizedSearchCV
10. Model Evaluation
11. Permutation Feature Importance
12. SHAP Model Explainability
13. Streamlit Deployment

## 🧹 Data Preprocessing

The preprocessing stage includes:

- Checking missing values
- Removing unnecessary features
- Converting appropriate data types
- Encoding categorical variables using one-hot encoding
- Scaling numerical features using `StandardScaler`
- Splitting the dataset into training and testing sets

The numerical features scaled in the project are:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`

The scaler is fitted using the **training data only** and then applied to the test data to prevent data leakage.

## 📊 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the relationship between customer characteristics and churn.

The analysis examined variables such as tenure, monthly charges, total charges, gender, contract type, internet service, payment method, online security, tech support, paperless billing, and other customer services.

## ⚖️ Handling Class Imbalance

The target variable was imbalanced, with significantly fewer churned customers than non-churned customers.

To address this problem, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training process.

For hyperparameter tuning, SMOTE was integrated into an `imblearn` pipeline so that oversampling occurs within the training folds during cross-validation.

## 🤖 Model Development

Three Random Forest approaches were evaluated:

### 1. Baseline Random Forest
A Random Forest classifier was first trained without oversampling or hyperparameter optimization.

### 2. Random Forest + SMOTE
SMOTE was applied to improve the model's ability to detect customers from the minority churn class.

### 3. Tuned Random Forest + SMOTE
The final approach combined SMOTE, Random Forest, RandomizedSearchCV, and 5-fold cross-validation.

## ⚙️ Hyperparameter Optimization

`RandomizedSearchCV` was used to search across different Random Forest configurations.

Parameters explored included:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `criterion`

The search used **5-fold cross-validation**, **F1-score** as the optimization metric, and **30 random parameter combinations**.

The selected parameters were:

```python
{
    'model__n_estimators': 300,
    'model__min_samples_split': 5,
    'model__min_samples_leaf': 2,
    'model__max_features': 'log2',
    'model__max_depth': 10,
    'model__criterion': 'gini'
}
```

## 📈 Model Evaluation

Because the dataset is imbalanced, model performance was evaluated using multiple metrics rather than accuracy alone.

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Baseline Random Forest | 0.816 | 0.594 |
| Random Forest + SMOTE | 0.815 | 0.588 |
| Tuned Random Forest + SMOTE | **0.836** | **0.634** |

> **Note:** These values are from the earlier model comparison. If the metrics changed after the final leakage-free retraining, update this table with the latest results.

## 🔍 Permutation Feature Importance

Permutation Importance was used to understand which features have the greatest impact on model performance.

Influential features identified during the analysis included internet service, tenure, contract type, total charges, payment method, paperless billing, online security, and monthly charges.

## 🧠 SHAP Model Explainability

SHAP (**SHapley Additive exPlanations**) was used to understand how different features influence the model's churn predictions.

The SHAP summary analysis helps explain:

- Which features have the greatest overall influence
- Whether feature values push predictions toward or away from churn
- How feature effects vary across customers

## 🌐 Streamlit Web Application

The trained model was integrated into an interactive web application using **Streamlit**.

Users can enter customer information and receive:

- Customer churn classification
- Predicted churn probability

## 🗂️ Repository Structure

```text
customer-churn-prediction-ml/
│
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── requirements.txt
├── CustomerChurnClassification.ipynb
└── README.md
```

## 💻 Run the Application Locally

Clone the repository:

```bash
git clone https://github.com/shathanaelmansour-collab/customer-churn-prediction-ml.git
```

Navigate to the project:

```bash
cd customer-churn-prediction-ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- SMOTE
- Random Forest
- RandomizedSearchCV
- SHAP
- Permutation Importance
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter / Google Colab

## 💡 Skills Demonstrated

- Exploratory Data Analysis
- Data Cleaning
- Feature Engineering
- Categorical Encoding
- Feature Scaling
- Classification
- Imbalanced Data Handling
- SMOTE
- Ensemble Machine Learning
- Hyperparameter Optimization
- Cross-Validation
- Model Evaluation
- Model Explainability
- SHAP
- Feature Importance
- Machine Learning Deployment
- Streamlit
- Git & GitHub

## 🚀 Future Improvements

- Compare additional classification algorithms such as XGBoost, LightGBM, and Logistic Regression
- Perform threshold optimization for churn classification
- Add SHAP explanations directly to the Streamlit application
- Improve the Streamlit interface and visualization
- Add automated preprocessing through a unified machine learning pipeline
- Deploy the application publicly for real-time predictions

## 👩‍💻 Author

**Shatha Mansour**

Data Science & Artificial Intelligence Graduate

Interested in Machine Learning, Data Analytics, Business Intelligence, and AI.

---

⭐ If you find this project useful, feel free to star the repository.
