import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Theme
sns.set_theme(style="whitegrid")

pd.set_option('display.max_columns', None)

plt.rcParams['figure.figsize'] = (12,6)
plt.rcParams['font.size'] = 12
df = pd.read_csv("payroll.csv", low_memory=False)

print("Dataset Loaded Successfully")
print("Duplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)
df['Agency Start Date'] = pd.to_datetime(
    df['Agency Start Date'],
    errors='coerce'
)
numeric_cols = [
    'Base Salary',
    'Regular Hours',
    'Regular Gross Paid',
    'OT Hours',
    'Total OT Paid',
    'Total Other Pay'
]

for col in numeric_cols:

    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('$', '', regex=False)
    )

    df[col] = pd.to_numeric(
        df[col],
        errors='coerce'
    )
    df['Employee_Name'] = (
    df['First Name'].astype(str)
    + ' '
    + df['Last Name'].astype(str)
)

current_year = pd.Timestamp.now().year

df['Years_of_Service'] = (
    current_year -
    df['Agency Start Date'].dt.year
)

df['Total_Compensation'] = (
    df['Regular Gross Paid']
    + df['Total OT Paid']
    + df['Total Other Pay']
)

df['OT_Percentage'] = np.where(
    df['Total_Compensation'] > 0,
    (df['Total OT Paid']/df['Total_Compensation'])*100,
    0
)
bins = [0,5,10,15,20,25,30,35,40,50]

labels = [
    '0-5',
    '6-10',
    '11-15',
    '16-20',
    '21-25',
    '26-30',
    '31-35',
    '36-40',
    '40+'
]

df['Service_Group'] = pd.cut(
    df['Years_of_Service'],
    bins=bins,
    labels=labels
)
plt.figure(figsize=(12,6))

sns.histplot(
    df['Base Salary'],
    bins=40,
    kde=True,
    color='royalblue'
)

plt.title('Base Salary Distribution', fontsize=18)

plt.show()
agency_payroll = (
    df.groupby('Agency Name')['Total_Compensation']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(14,7))

sns.barplot(
    x=agency_payroll.index,
    y=agency_payroll.values,
    palette='viridis'
)

plt.xticks(rotation=45)

plt.title("Top 10 Agencies by Payroll")

plt.show()
top_ot = (
    df.groupby('Agency Name')['Total OT Paid']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,7))

sns.barplot(
    x=top_ot.values,
    y=top_ot.index,
    palette='rocket'
)

plt.title('Top Agencies by OT Payment')

plt.show()
top20 = (
    df[['Employee_Name','Total_Compensation']]
    .sort_values(
        by='Total_Compensation',
        ascending=False
    )
    .head(20)
)

plt.figure(figsize=(14,10))

sns.barplot(
    x='Total_Compensation',
    y='Employee_Name',
    data=top20,
    palette='mako'
)

plt.title('Top 20 Highest Paid Employees')

plt.show()
borough_payroll = (
    df.groupby(
        'Work Location Borough'
    )['Total_Compensation']
    .sum()
    .sort_values()
)

plt.figure(figsize=(12,8))

sns.barplot(
    x=borough_payroll.values,
    y=borough_payroll.index,
    palette='Set2'
)

plt.title('Payroll by Borough')

plt.show()
corr_cols = [
    'Base Salary',
    'Regular Hours',
    'Regular Gross Paid',
    'OT Hours',
    'Total OT Paid',
    'Total Other Pay',
    'Years_of_Service',
    'Total_Compensation',
    'OT_Percentage'
]

plt.figure(figsize=(12,8))

sns.heatmap(
    df[corr_cols].corr(),
    annot=True,
    cmap='RdYlBu',
    fmt='.2f',
    linewidths=1
)

plt.title('Correlation Matrix')

plt.show()
plt.figure(figsize=(12,7))

sns.scatterplot(
    data=df,
    x='OT Hours',
    y='Total_Compensation',
    hue='Years_of_Service',
    palette='plasma',
    alpha=0.7
)

plt.title('OT Hours vs Total Compensation')

plt.show()
plt.figure(figsize=(12,4))

sns.boxplot(
    x=df['Total_Compensation'],
    color='gold'
)

plt.title('Total Compensation Outliers')

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    x='Service_Group',
    palette='husl'
)

plt.title('Employee Distribution by Years of Service')

plt.xlabel('Years of Service')

plt.ylabel('Employee Count')

plt.show()
# ----------------------------------------
# Feature Engineering
# ----------------------------------------

df['Employee_Name'] = (
    df['First Name'].astype(str)
    + ' '
    + df['Last Name'].astype(str)
)

current_year = pd.Timestamp.now().year

df['Years_of_Service'] = (
    current_year -
    df['Agency Start Date'].dt.year
)

df['Total_Compensation'] = (
    df['Regular Gross Paid']
    + df['Total OT Paid']
    + df['Total Other Pay']
)

df['OT_Percentage'] = np.where(
    df['Total_Compensation'] > 0,
    (df['Total OT Paid']/df['Total_Compensation'])*100,
    0
)

# ----------------------------------------
# IQR OUTLIER DETECTION
# ----------------------------------------

Q1 = df['Total_Compensation'].quantile(0.25)
Q3 = df['Total_Compensation'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df['Total_Compensation'] < lower_bound) |
    (df['Total_Compensation'] > upper_bound)
]

print("\n========== OUTLIER ANALYSIS ==========")
print("Number of outliers:", len(outliers))

print(
    outliers[
        [
            'Employee_Name',
            'Agency Name',
            'Total_Compensation'
        ]
    ]
    .sort_values(
        by='Total_Compensation',
        ascending=False
    )
    .head(20)
)


# ----------------------------------------
# IQR OUTLIER DETECTION
# ----------------------------------------

Q1 = df['Total_Compensation'].quantile(0.25)
Q3 = df['Total_Compensation'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df['Total_Compensation'] < lower_bound) |
    (df['Total_Compensation'] > upper_bound)
]

print("\n========== IQR OUTLIERS ==========")
print("Number of outliers:", len(outliers))

print(
    outliers[
        [
            'Employee_Name',
            'Agency Name',
            'Total_Compensation'
        ]
    ]
    .sort_values(
        by='Total_Compensation',
        ascending=False
    )
    .head(20)
)

plt.figure(figsize=(14,5))

sns.boxplot(
    x=df['Total_Compensation'],
    color='gold'
)

plt.title("Compensation Outliers")

plt.show()

# ----------------------------------------
# Z-SCORE OUTLIERS
# ----------------------------------------

df['Z_Score'] = zscore(df['Total_Compensation'])

z_outliers = df[
    abs(df['Z_Score']) > 3
]

print("\n========== Z-SCORE OUTLIERS ==========")

print(
    z_outliers[
        [
            'Employee_Name',
            'Agency Name',
            'Total_Compensation',
            'Z_Score'
        ]
    ]
    .sort_values(
        by='Total_Compensation',
        ascending=False
    )
    .head(20)
)

plt.figure(figsize=(14,7))

sns.scatterplot(
    data=df,
    x=df.index,
    y='Total_Compensation',
    color='skyblue',
    alpha=0.4
)

sns.scatterplot(
    data=z_outliers,
    x=z_outliers.index,
    y='Total_Compensation',
    color='red',
    s=100,
    label='Outliers'
)

plt.title("Z-Score Outlier Detection")

plt.xlabel("Employee Index")

plt.ylabel("Total Compensation")

plt.show()

# ----------------------------------------
# ISOLATION FOREST
# ----------------------------------------

iso_model = IsolationForest(
    contamination=0.01,
    random_state=42
)

df['Anomaly'] = iso_model.fit_predict(
    df[['Total_Compensation']]
)
anomalies = df[
    df['Anomaly'] == -1
]

print("\n========== ISOLATION FOREST ANOMALIES ==========")

print(
    anomalies[
        [
            'Employee_Name',
            'Agency Name',
            'Total_Compensation'
        ]
    ]
    .sort_values(
        by='Total_Compensation',
        ascending=False
    )
    .head(20)
)
plt.figure(figsize=(14,7))

sns.scatterplot(
    data=df,
    x=df.index,
    y='Total_Compensation',
    hue='Anomaly',
    palette='Set1',
    alpha=0.7
)

plt.title("Isolation Forest Anomaly Detection")

plt.xlabel("Employee Index")

plt.ylabel("Total Compensation")

plt.show()
# ----------------------------------------
# EMPLOYEE SEGMENTATION
# ----------------------------------------

features = df[
    [
        'Base Salary',
        'OT Hours',
        'Total OT Paid',
        'Years_of_Service'
    ]
]

features = features.fillna(0)
scaler = StandardScaler()

X_scaled = scaler.fit_transform(features)
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X_scaled)
print("\n========== CLUSTER COUNTS ==========")

print(df['Cluster'].value_counts())
cluster_summary = (
    df.groupby('Cluster')
    [
        [
            'Base Salary',
            'OT Hours',
            'Total OT Paid',
            'Years_of_Service',
            'Total_Compensation'
        ]
    ]
    .mean()
)

print("\n========== CLUSTER SUMMARY ==========")

print(cluster_summary)
plt.figure(figsize=(12,8))

sns.scatterplot(
    data=df,
    x='Base Salary',
    y='Total_Compensation',
    hue='Cluster',
    palette='tab10',
    alpha=0.7
)

plt.title(
    'Employee Segmentation: Salary vs Compensation',
    fontsize=18
)

plt.show()
plt.figure(figsize=(12,8))

sns.scatterplot(
    data=df,
    x='OT Hours',
    y='Total OT Paid',
    hue='Cluster',
    palette='Set2',
    alpha=0.7
)

plt.title(
    'Employee Clusters by Overtime Behavior',
    fontsize=18
)

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    x='Cluster',
    palette='viridis'
)

plt.title(
    'Employees per Cluster',
    fontsize=18
)

plt.show()
# ----------------------------------------
# SALARY PREDICTION
# ----------------------------------------

X = df[
    [
        'Base Salary',
        'OT Hours',
        'Total OT Paid',
        'Years_of_Service'
    ]
]

y = df['Total_Compensation']
X = X.fillna(0)
y = y.fillna(0)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)
predictions = rf.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE : {mae:,.2f}")

print(f"RMSE : {rmse:,.2f}")

print(f"R² Score : {r2:.4f}")
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

importance = importance.sort_values()
plt.figure(figsize=(10,6))

sns.barplot(
    x=importance.values,
    y=importance.index,
    palette='viridis'
)

plt.title(
    'Feature Importance',
    fontsize=18
)

plt.xlabel(
    'Importance Score'
)

plt.show()
plt.figure(figsize=(10,8))

plt.scatter(
    y_test,
    predictions,
    alpha=0.4,
    color='royalblue'
)

plt.xlabel(
    'Actual Compensation'
)

plt.ylabel(
    'Predicted Compensation'
)

plt.title(
    'Actual vs Predicted Compensation',
    fontsize=18
)

plt.show()
df.to_csv("payroll_dashboard_data.csv", index=False)

print("Dashboard file exported successfully!")