import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, jsonify

#These lines import necessary libraries and modules for the project.NumPy (np) and pandas (pd) are used for data manipulation and analysis


app = Flask(__name__)

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

#This section prepares the data for model training. It selects the features ('Melting Point(K)', 'Boiling Point (K)') as input (X) and the target variable ('Element') as output (y). It also loads atomic numbers into a dictionary.




X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#splits the data into training set and testing set



# It's often a good idea to scale your input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)  
clf.fit(X_train, y_train)

#This section creates a Random Forest Classifier model with 100 estimators and trains it using the training data (X_train, y_train).


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        melting_point = float(request.form['melting_point'])
        boiling_point = float(request.form['boiling_point'])

        # Create a DataFrame for the new element
        new_element = pd.DataFrame([[melting_point, boiling_point]], columns=['Melting Point(K)', 'Boiling Point (K)'])

        # scale the new data as the model was trained with scaled data
        new_element = scaler.transform(new_element)

        # Use the model to predict the element
        predicted_element = clf.predict(new_element)

        response_data = {'predicted_element': predicted_element[0], 'atomic_number': element_to_atomic_number[predicted_element[0]]}
        
        try:
            return jsonify(response_data)
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)