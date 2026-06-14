# 🏛️ Payroll Analytics 

## 📌 Overview

This project performs an end-to-end analysis of U.S. payroll data to uncover compensation trends, overtime patterns, employee segmentation, and anomalous payroll behavior.

Using Python, statistical analysis, and machine learning techniques, the project provides business insights that can help organizations optimize payroll spending and identify unusual compensation patterns.

---

## 🎯 Objectives

The goal of this project is to:

* Analyze payroll spending across agencies and boroughs.
* Identify the highest-paid agencies and employees.
* Understand overtime behavior and compensation drivers.
* Detect unusual payroll patterns using statistical methods and machine learning.
* Segment employees based on compensation and experience.
* Predict employee total compensation using a machine learning model.

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* SciPy

---

# 📂 Dataset Features

The dataset contains information about employee payroll records, including:

* Fiscal Year
* Agency Name
* Last Name
* First Name
* Agency Start Date
* Work Location Borough
* Title Description
* Leave Status as of June 30
* Base Salary
* Pay Basis
* Regular Hours
* Regular Gross Paid
* OT Hours
* Total OT Paid
* Total Other Pay

---

# 🔄 Project Workflow

## 1. Data Cleaning

Performed:

* Removed duplicate records
* Converted date columns
* Converted salary columns to numeric values
* Handled missing values

---

## 2. Feature Engineering

Created the following features:

* Employee_Name
* Years_of_Service
* Total_Compensation
* OT_Percentage
* Service_Group

---

## 3. Exploratory Data Analysis

Conducted analysis to answer questions such as:

* Which agencies spend the most?
* Which employees receive the highest compensation?
* How is payroll distributed across boroughs?
* How does overtime affect compensation?

---

## 4. Business Insights

Generated insights regarding:

* Payroll spending patterns
* Overtime expenses
* Workforce experience
* Compensation distribution
* Agency-level payroll analysis

---

## 5. Outlier Detection

Implemented multiple techniques:

### IQR Method

Used to detect unusually high compensation values.

### Z-Score Method

Identified extreme compensation records using standard deviation.

### Isolation Forest

Machine learning-based anomaly detection for identifying suspicious payroll behavior.

---

## 6. Employee Segmentation

Applied K-Means clustering using:

* Base Salary
* OT Hours
* Total OT Paid
* Years of Service

This grouped employees into clusters based on compensation and overtime behavior.

---

## 7. Salary Prediction

Built a Random Forest Regression model to predict:

### Target Variable

* Total Compensation

### Features Used

* Base Salary
* OT Hours
* Total OT Paid
* Years of Service

---

# 📊 Model Performance

| Metric   | Value     |
| -------- | --------- |
| MAE      | 7,036.19  |
| RMSE     | 13,880.70 |
| R² Score | 0.9008    |

The model explains approximately **90% of the variance** in employee compensation, demonstrating strong predictive performance.

---

# 📈 Visualizations

## Salary Analysis

* Base Salary Distribution
* Top 10 Agencies by Payroll
* Top Agencies by OT Payments
* Top 20 Highest Paid Employees
* Payroll by Borough

---

## Correlation and Compensation Analysis

* Correlation Matrix
* OT Hours vs Total Compensation
* Total Compensation Outliers
* Employee Distribution by Years of Service

---

## Outlier and Anomaly Detection

* Compensation Outliers
* Z-Score Outlier Detection
* Isolation Forest Anomaly Detection

---

## Employee Segmentation

* Employee Segmentation: Salary vs Compensation
* Employee Clusters by Overtime Behaviour
* Employees per Cluster

---

## Machine Learning Evaluation

* Feature Importance
* Actual vs Predicted Compensation

---

# 📁 Project Structure

```text
payroll-analytics-ml-project
│
├── data
│     payroll.csv
│
├── outputs
│     anomalies.csv
│     clusters.csv
│     predictions.csv
│
├── images
│     base_salary_distribution.png
│     top_10_agencies_payroll.png
│     top_ot_agencies.png
│     top_20_highest_paid_employees.png
│     payroll_by_borough.png
│     correlation_matrix.png
│     ot_hours_vs_total_compensation.png
│     total_compensation_outliers.png
│     employee_distribution_years_service.png
│     compensation_outliers.png
│     z_score_outlier_detection.png
│     isolation_forest_anomaly_detection.png
│     employee_segmentation_salary_vs_compensation.png
│     employee_clusters_overtime_behavior.png
│     employees_per_cluster.png
│     feature_importance.png
│     actual_vs_predicted_compensation.png
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scipy
sklearn
```

---

# 📌 Key Skills Demonstrated

### Data Cleaning

* Duplicate removal
* Data type conversion
* Missing value handling

### Feature Engineering

* Derived variables
* Compensation metrics
* Service categorization

### Exploratory Data Analysis

* Distribution analysis
* Correlation analysis
* Business insights generation

### Data Visualization

* Histograms
* Bar charts
* Scatter plots
* Heatmaps
* Boxplots

### Statistical Analysis

* IQR Outlier Detection
* Z-Score Analysis

### Machine Learning

* Isolation Forest Anomaly Detection
* K-Means Clustering
* Random Forest Regression

### Model Evaluation

* MAE
* RMSE
* R² Score
* Feature Importance Analysis

---

# 🚀 Future Improvements

* Interactive Power BI Dashboard
* Streamlit Web Application
* XGBoost Regression
* SHAP Explainability
* Payroll Forecasting
* Model Deployment

---

# ⭐ Conclusion

This project demonstrates an end-to-end data analytics and machine learning workflow, transforming raw payroll data into actionable business insights and predictive models.

The project combines data cleaning, exploratory analysis, anomaly detection, clustering, and regression modeling to provide a comprehensive payroll analytics solution.

---

## 👨‍💻 Author

**Indra Kumar**

Data Analytics | Machine Learning | Python | SQL | Power BI

---
