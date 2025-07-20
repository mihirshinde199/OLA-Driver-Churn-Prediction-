# OLA-Driver-Churn-Prediction-

# 🚗 OLA Driver Churn Prediction

This project tackles a real-world business problem faced by Ola Cabs — **high driver attrition**. The objective is to build a machine learning model to predict whether a driver is likely to leave the platform, helping Ola improve retention strategies and reduce hiring costs.

## 📊 Problem Statement

Ola is experiencing high churn among its drivers, which negatively impacts operational efficiency and increases costs. Using historical data from 2019–2020, this project builds predictive models using ensemble learning techniques to identify which drivers are at risk of leaving.

## 🧠 Solution Approach

The entire workflow follows a robust data science pipeline:

- **Data Understanding and Cleaning**
- **Exploratory Data Analysis (EDA)**
- **KNN Imputation for missing values**
- **Feature Engineering**
- **Class Imbalance Handling (SMOTE)**
- **Standardization and Encoding**
- **Modeling using Ensemble Learning (Bagging & Boosting)**
- **Evaluation via ROC-AUC, Confusion Matrix, Classification Report**
- **Actionable Business Insights**

## 📁 Dataset Overview

| Column               | Description |
|----------------------|-------------|
| Driver_ID            | Unique ID of the driver |
| Age, Gender, City    | Demographics |
| Education_Level      | 10+, 12+, Graduate |
| Income               | Monthly average income |
| Date Of Joining      | Joining date of driver |
| LastWorkingDate      | Last working date (used to derive churn) |
| Quarterly Rating     | Rating between 1–5 |
| Total Business Value | Net business value per month |

## 🧪 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, RandomForestClassifier
- Matplotlib, Seaborn
- Imblearn (for SMOTE)
- Jupyter Notebook

## 🛠️ Models Trained

- ✅ Random Forest (Bagging)
- ✅ XGBoost Classifier (Boosting)

## 📈 Evaluation Metrics

- ROC AUC Curve
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix

## 🔍 Key Insights

- Drivers with low ratings and decreasing income tend to churn more often.
- Education level and city also play a role in retention probability.
- Improving driver incentives for mid-tier performers could reduce attrition.

## 📝 How to Run

1. Clone the repository:
git clone https://github.com/mihirshinde199/OLA-Driver-Churn-Prediction 

2. Navigate to the directory and open the Jupyter notebook:2. Navigate to the directory and open the Jupyter notebook:
jupyter notebook OLA_Driver_Churn_Prediction.ipynb

3. Install dependencies (if required):
pip install -r requirements.txt


## 👨‍💻 Author

**Mihir Shinde**  
📧 mihir.shinde18@vit.edu
🔗 [LinkedIn Profile](https://www.linkedin.com/in/yourusername)  
🐙 [GitHub](https://github.com/mihirshinde199)

---

⭐ If you found this useful, give it a star!
