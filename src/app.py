import streamlit as st
from pickle import load
import pandas as pd

pipeline = load(open("trained_pipeline.pkl", "rb"))

st.title("Iris - Model prediction")

val1 = st.slider("Sepel length (cm)", min_value = 0.0, max_value = 4.0, step = 0.1)
val2 = st.slider("Sepel width (cm)", min_value = 0.0, max_value = 4.0, step = 0.1)
val3 = st.slider("Petal legnth (cm)", min_value = 0.0, max_value = 4.0, step = 0.1)
val4 = st.slider("Petal width (cm)", min_value = 0.0, max_value = 4.0, step = 0.1)

if st.button("Predict"):
    # New data point for prediction (example with some missing values)
    new_data_point_dict = {
        'sepal length (cm)': [val1],
        'sepal width (cm)': [val2],
        'petal length (cm)': [val3],
        'petal width (cm)': [val4]
    }

    # Convert the dictionary to a DataFrame
    new_data_point_df = pd.DataFrame(new_data_point_dict)

    # Pipeline makes a class prediction
    pred_class = pipeline.predict(new_data_point_df)[0]
    st.write("Prediction:", pred_class)

