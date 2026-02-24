# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from numpy.linalg import norm

# Load Dataset
data = pd.read_csv("student_dataset.csv")

print("Dataset Loaded Successfully!\n")
print("First 5 Rows:\n")
print(data.head())

print("\=== STEP 1 ===\n")

mean_math = data["Math_Score"].mean()
median_math = data["Math_Score"].median()
mode_math = data["Math_Score"].mode()[0]

print("Math Score:")
print("Mean:", mean_math)
print("Median:", median_math)
print("Mode:", mode_math)

range_science = data["Science_Score"].max() - data["Science_Score"].min()
variance_science = data["Science_Score"].var()
std_science = data["Science_Score"].std()

print("\nScience Score:")
print("Range:", range_science)
print("Variance:", variance_science)
print("Standard Deviation:", std_science)

print("\n=== STEP 2 ===\n")

prob_pass = data["Pass_Fail"].mean()
print("Probability of Passing:", prob_pass)

contingency = pd.crosstab(data["Pass_Fail"], data["Hours_Studied"] > 5)
print("\nContingency Table (Pass vs Hours_Studied > 5):")
print(contingency)

cond_prob = len(data[(data["Pass_Fail"] == 1) & 
                     (data["Hours_Studied"] > 5)]) / len(data[data["Hours_Studied"] > 5])

print("\nP(Pass | Hours_Studied > 5):", cond_prob)

print("\n=== STEP 3 ===\n")

plt.figure()
sns.histplot(data["Math_Score"], kde=True)
plt.title("Math Score Distribution")
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.show()

skew_science = stats.skew(data["Science_Score"])
kurt_science = stats.kurtosis(data["Science_Score"])

print("Science Score Skewness:", skew_science)
print("Science Score Kurtosis:", kurt_science)

plt.figure()
stats.probplot(data["English_Score"], dist="norm", plot=plt)
plt.title("Q-Q Plot for English Score")
plt.show()

print("\=== STEP 4 ===\n")

math_vector = data["Math_Score"].head(5).values
science_vector = data["Science_Score"].head(5).values

print("Math Vector:", math_vector)
print("Science Vector:", science_vector)

dot_product = np.dot(math_vector, science_vector)
print("\nDot Product:", dot_product)

norm_math = norm(math_vector)
norm_science = norm(science_vector)

print("Norm of Math Vector:", norm_math)
print("Norm of Science Vector:", norm_science)

angle = np.arccos(dot_product / (norm_math * norm_science))
angle_degrees = np.degrees(angle)

print("Angle between vectors (degrees):", angle_degrees)

print("\n=== PROJECT COMPLETED SUCCESSFULLY ===")