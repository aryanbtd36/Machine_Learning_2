#LOGICAL REGRESSION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = {
    'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj', 'Ayush', 'Riyaz', 'Priya', 'Yuvraj', 'Aditya'], 
    'Age': [17, 18, 20, 18, 21, 22, 19, 23, 24],
    'Address': ['Delhi', 'Kanpur', 'Prayagraj', 'Varanasi', 'Kannauj', 'Mumbai', 'Pune', 'Nagpur', 'Chennai'],
    'Qualification': ['Msc', 'MA', 'MCA', 'Phd', 'Btech', 'Mtech', 'BBA', 'MBA', 'BCA'],
    'Salaries': [70000, 50000, 65000, 100000, 150000, 200000, 90000, 95000, 110000],
}
df = pd.DataFrame(data)
threshold = 100000
df['HighSalary'] = np.where(df['Salaries'] > threshold, 1, 0)  # 1 for High Salary, 0 for Low Salary
X = df[['Age']].values  # Independent variable
y = df['HighSalary'].values  # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data (0=Low, 1=High)')
plt.scatter(X_test, y_test, color='red', label='Test data (0=Low, 1=High')
plt.scatter(X_test, y_pred, color='green', label='Predictions', marker='x')
age_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
probabilities = model.predict_proba(age_range)[:, 1]  # Probability of High Salary
plt.plot(age_range, probabilities, color='orange', label='Logistic Regression Line', linewidth=2)
plt.title('Logistic Regression: High Salary vs Age')
plt.xlabel('Age')
plt.ylabel('Probability of High Salary (1 = Yes, 0 = No)')
plt.axhline(0.5, color='gray', linestyle='--')  # Decision boundary
plt.legend()
plt.show()
df.to_csv('saket.csv', index=False)  # index=False to avoid saving the index column
print("\nDataFrame saved to 'saket.csv'")