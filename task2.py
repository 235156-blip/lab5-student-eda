import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Create synthetic dataset with 12 students
data = {
    "StudentID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 111],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Male"],
    "StudyHours": [2.5, 5.0, 3.0, 7.5, 1.5, 6.0, 4.0, 8.0, 3.5, 6.5, 2.0, 2.0],
    "Attendance(%)": [80, 92, 75, 95, 70, 88, 85, 98, 82, 90, 65, 65],
    "Marks": [65.0, 85.0, np.nan, 92.0, 50.0, np.nan, 72.0, 96.0, 68.0, 89.0, 58.0, 58.0]
}

# Write CSV and load back
pd.DataFrame(data).to_csv("student_performance.csv", index=False)
df = pd.read_csv("student_performance.csv")

# (a) Inspection
print("Shape:", df.shape)
print(df.info())

# (b) Missing & Duplicate values
print("\nMissing values in Marks:", df["Marks"].isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Clean data
df_clean = df.drop_duplicates().copy()
median_marks = df_clean["Marks"].median()
df_clean["Marks"] = df_clean["Marks"].fillna(median_marks)
print("After cleaning shape:", df_clean.shape)

# (c) Histograms and Boxplots
plt.figure(figsize=(6, 4))
plt.hist(df_clean["Marks"], bins=5, color="steelblue", edgecolor="black")
plt.title("Marks Distribution")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("marks_histogram.png")
plt.close()

plt.figure(figsize=(4, 5))
plt.boxplot(df_clean["Marks"])
plt.title("Marks Boxplot")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("marks_boxplot.png")
plt.close()

# (d) Scatter plot & Correlation matrix
plt.figure(figsize=(6, 4))
plt.scatter(df_clean["StudyHours"], df_clean["Marks"], color="darkorange")
plt.xlabel("StudyHours")
plt.ylabel("Marks")
plt.title("StudyHours vs. Marks")
plt.tight_layout()
plt.savefig("studyhours_vs_marks.png")
plt.close()

print("\nCorrelation matrix:")
print(df_clean[["StudyHours", "Attendance(%)", "Marks"]].corr())

# (e) Categorical comparison
print("\nAverage Marks and Attendance by Gender:")
print(df_clean.groupby("Gender")[["Marks", "Attendance(%)"]].mean())

print("\nSaved marks_histogram.png, marks_boxplot.png, studyhours_vs_marks.png")
