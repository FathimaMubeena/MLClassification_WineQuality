import streamlit as st
import pandas as pd
import joblib

# Load the trained model (pipeline)
model = joblib.load('wine_quality_predictor_model.pkl')

# Title of the web app
st.title("Wine Quality Prediction multi-label classification")
st.subheader("Wine Quality  :  low/0 , medium/1, high/2 ")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Function to get UserInput and return Dataframe of features
def user_input_features():

    fixed_acidity= st.sidebar.number_input('Enter a number for fixed_acidity')
    volatile_acidity= st.sidebar.number_input('Enter a number for volatile acidity')
    citric_acid= st.sidebar.number_input('Enter a number for citric acid')
    residual_sugar= st.sidebar.number_input('Enter a number for residual sugar')
    chlorides= st.sidebar.number_input('Enter a number for chlorides')
    free_sulfur_dioxide= st.sidebar.number_input('Enter a number for free sulfur dioxide')
    total_sulfur_dioxide= st.sidebar.number_input('Enter a number for total sulfur dioxide')
    density= st.sidebar.number_input('Enter a number for density')
    pH= st.sidebar.number_input('Enter a number for pH')
    sulphates= st.sidebar.number_input('Enter a number for sulphates')
    alcohol= st.sidebar.number_input('Enter a number for alcohol')


    sample_input_dictionary = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }

    features_df = pd.DataFrame(sample_input_dictionary, index=[0])
    return features_df

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input parameters')
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)

# Display predictions
st.subheader('Prediction')
st.write(f'Predicted Class of WineQuality: {int(prediction[0])}')