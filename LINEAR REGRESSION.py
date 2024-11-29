#LINEAR REGRESSION 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create a dictionary with data
data = {
    'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj', 'Ayush', 'Riyaz', 'Priya', 'Yuvraj', 'Aditya'], 
    'Age': [17, 18, 20, 18, 21, 22, 19, 23, 24],
    'Address': ['Delhi', 'Kanpur', 'Prayagraj', 'Varanasi', 'Kannauj', 'Mumbai', 'Pune', 'Nagpur', 'Chennai'],
    'Qualification': ['Msc', 'MA', 'MCA', 'Phd', 'Btech', 'Mtech', 'BBA', 'MBA', 'BCA'],
    'Salaries': [70000, 50000, 65000, 100000, 150000, 200000, 90000, 95000, 110000],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Prepare the data for linear regression
X = df[['Age']].values  # Independent variable
y = df['Salaries'].values  # Dependent variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
r_squared = r2_score(y_test, y_pred)
print('R-squared:', r_squared)

# Visualize the results
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='red', label='Test data')
plt.plot(X_test, y_pred, color='green', label='Regression line')
plt.title('Linear Regression: Salaries vs Age')
plt.xlabel('Age')
plt.ylabel('Salaries')
plt.legend()
plt.show()

# Save the DataFrame to a CSV file
df.to_csv('saket.csv', index=False)  # index=False to avoid saving the index column
print("\nDataFrame saved to 'saket.csv'")