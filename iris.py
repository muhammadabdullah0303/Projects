import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load Model 
dt_model = pickle.load(open('./decision_tree_model.pkl', 'rb'))

# Define a function to predict iris species
def predict_species(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
    features = [[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]]
    return dt_model.predict(features)

# Streamlit UI
st.title('Iris Species Prediction Random Forest')

# Add sliders for user input
SepalLengthCm = st.slider('SepalLengthCm', min_value=0.0, max_value=10.0, step=0.1)
SepalWidthCm = st.slider('SepalWidthCm', min_value=0.0, max_value=10.0, step=0.1)
PetalLengthCm = st.slider('PetalLengthCm', min_value=0.0, max_value=10.0, step=0.1)
PetalWidthCm = st.slider('PetalWidthCm', min_value=0.0, max_value=10.0, step=0.1)

# # Add Text for user input
# sepal_length = st.text_input('Sepal Length')
# sepal_width = st.text_input('Sepal Width',)
# petal_length = st.text_input('Petal Length',)
# petal_width = st.text_input('Petal Width', )

if st.button('Predict'):
    prediction = predict_species(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
    species = [prediction][0]
    st.write(f'Predicted Species: {species}')

    if prediction == 'Iris-setosa':
        st.image('setosa.jpg')
    elif prediction == 'Iris-versicolor':
        st.image('versicolor.jpg')
    elif prediction == 'Iris-virginica':
        st.image('virginica.jpg')
