import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read CSV file
data = pd.read_csv("student_data.csv")

print("Columns found:")
print(data.columns)

print(data.head())

# Features (Hours Studied)
X = data[["hours_studied"]]

# Target (Exam Score)
y = data["exam_score"]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Take input from user
hours = float(input("Enter hours studied: "))

# Predict marks
predicted_score = model.predict([[hours]])

print(f"Predicted Exam Score: {predicted_score[0]:.2f}")

# Plot data points
plt.scatter(X, y, label="Actual Data")

# Plot regression line
plt.plot(X, model.predict(X), label="Regression Line")

plt.xlabel("hours_studied")
plt.ylabel("exam_score")
plt.title("Hours Studied vs Exam Score")
plt.legend()

plt.show()
