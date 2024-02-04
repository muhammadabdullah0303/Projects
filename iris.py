import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a simple RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Define a function to predict iris species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    return model.predict(features)

# Streamlit UI
st.title('Iris Species Prediction Random Forest')

# Add sliders for user input
sepal_length = st.slider('Sepal Length', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.slider('Sepal Width', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.slider('Petal Length', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.slider('Petal Width', min_value=0.0, max_value=10.0, step=0.1)

# # Add Text for user input
# sepal_length = st.text_input('Sepal Length')
# sepal_width = st.text_input('Sepal Width',)
# petal_length = st.text_input('Petal Length',)
# petal_width = st.text_input('Petal Width', )

if st.button('Predict'):
    prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    species = iris.target_names[prediction][0]
    st.write(f'Predicted Species: {species}')

    if prediction == 'setosa':
        st.image('setosa.jpg')
    elif prediction == 'versicolor':
        st.image('versicolor.jpg')
    elif prediction == 'virginica':
        st.image('virginica.jpg')
