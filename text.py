import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_csv('elements.csv')

# Replace '-' with NaN and then fill NaNs with the median of the column
data = data.replace('-', np.nan)
data['Boiling Point (K)'] = pd.to_numeric(data['Boiling Point (K)'])
data['Boiling Point (K)'] = data['Boiling Point (K)'].fillna(data['Boiling Point (K)'].median())

# Prepare the data
X = data[['Melting Point(K)', 'Boiling Point (K)']]
y = data['Element']
atomic_numbers = data['Atomic Number']  # Load the atomic numbers

# Create a dictionary mapping elements to atomic numbers
element_to_atomic_number = dict(zip(y, atomic_numbers))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# It's often a good idea to scale your input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the ANN
clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=2000, alpha=0.001, random_state=42)
clf.fit(X_train, y_train)

# Test the ANN
print("Training score: ", clf.score(X_train, y_train))
print("Testing score: ", clf.score(X_test, y_test))

# Ask the user to input the melting point and boiling point
melting_point = float(input("Please enter the melting point: "))
boiling_point = float(input("Please enter the boiling point: "))

# Create a DataFrame for the new element
new_element = pd.DataFrame([[melting_point, boiling_point]], columns=['Melting Point(K)', 'Boiling Point (K)'])

# Don't forget to scale the new data as the model was trained with scaled data
new_element = scaler.transform(new_element)

# Use the model to predict the element
predicted_element = clf.predict(new_element)

print("The predicted element is: ", predicted_element)
print("The atomic number of the predicted element is: ", element_to_atomic_number[predicted_element[0]])
