
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("employee_performance.csv")

df = df.drop_duplicates()

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Experience"] = df["Experience"].fillna(df["Experience"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["Performance_Score"] = df["Performance_Score"].fillna(df["Performance_Score"].median())
df["Satisfaction"] = df["Satisfaction"].fillna(df["Satisfaction"].median())
df["Projects_Completed"] = df["Projects_Completed"].fillna(df["Projects_Completed"].median())

df["Performance_Level"] = np.select(
    [df["Performance_Score"] >= 90, df["Performance_Score"] >= 80],
    ["Excellent", "Good"],
    default="Needs Improvement"
)

print("Dataset Overview")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nTotal Employees:", df["Employee_ID"].nunique())
print("Average Salary:", round(df["Salary"].mean(), 2))
print("Average Performance:", round(df["Performance_Score"].mean(), 2))
print("Average Satisfaction:", round(df["Satisfaction"].mean(), 2))

department_performance = df.groupby("Department")["Performance_Score"].mean().sort_values(ascending=False)
print("\nDepartment-wise Performance")
print(department_performance)

department_salary = df.groupby("Department")["Salary"].mean().sort_values(ascending=False)
print("\nDepartment-wise Average Salary")
print(department_salary)

performance_levels = df["Performance_Level"].value_counts()
print("\nPerformance Level Distribution")
print(performance_levels)

gender_analysis = df.groupby("Gender")["Performance_Score"].mean()
print("\nGender-wise Average Performance")
print(gender_analysis)

experience_analysis = df.groupby("Experience")["Performance_Score"].mean()
print("\nExperience-wise Performance")
print(experience_analysis)

project_analysis = df.groupby("Department")["Projects_Completed"].mean()
print("\nDepartment-wise Projects Completed")
print(project_analysis)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 5))
sns.barplot(x=department_performance.index, y=department_performance.values)
plt.title("Department-wise Average Performance")
plt.xlabel("Department")
plt.ylabel("Performance Score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.barplot(x=department_salary.index, y=department_salary.values)
plt.title("Department-wise Average Salary")
plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Performance_Level")
plt.title("Employee Performance Distribution")
plt.xlabel("Performance Level")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Experience", y="Performance_Score", hue="Department")
plt.title("Experience vs Performance")
plt.xlabel("Experience in Years")
plt.ylabel("Performance Score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Department", y="Salary")
plt.title("Department-wise Salary Distribution")
plt.xlabel("Department")
plt.ylabel("Salary")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[["Age", "Experience", "Salary", "Performance_Score", "Satisfaction", "Projects_Completed"]].corr(), annot=True, cmap="coolwarm")
plt.title("Employee Data Correlation")
plt.tight_layout()
plt.show()

print("\nBusiness Insights")
print("Best Performing Department:", department_performance.idxmax())
print("Highest Average Salary Department:", department_salary.idxmax())
print("Most Common Performance Level:", performance_levels.idxmax())
print("Highest Average Performance by Gender:", gender_analysis.idxmax())
print("Department with Most Projects:", project_analysis.idxmax())
