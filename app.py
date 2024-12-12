from __future__ import annotations
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> ArrayLike:
    # Load the model from the file
    try:
        model = load(filename)
    except FileNotFoundError:
        st.error(f"Model file '{filename}' not found.")
        return np.array([None])
    # Make predictions
    y = model.predict(X)
    return y

def create_streamlit_app():
    st.title("Linear Regression Prediction App")
    st.write("Move the slider to select a feature value and click the 'Predict value' button to see predictions.")
    
    input_feature = st.slider("Input Feature Value", min_value=-3.0, max_value=3.0, step=0.1)

    if st.button("Predict value"):
        st.write("Button clicked. Predicting...")
        prediction = load_and_predict(np.array([[input_feature]]))
        if prediction[0] is not None:
            st.write(f"Predicted Target Value: {prediction[0]}")
            visualize_difference(input_feature, prediction[0])
        else:
            st.write("Prediction could not be completed.")

def visualize_difference(input_feature: float, prediction: ArrayLike):
    try:
        X = load("X.joblib")
        y = load("y.joblib")
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return

    actual_target = y[_index_of_closest(X, input_feature)]
    difference = actual_target - prediction
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(X, y, color="gray", alpha=0.5, label="Dataset")
    plt.scatter(input_feature, actual_target, color="blue", label="Actual Target")
    plt.scatter(input_feature, prediction, color="red", label="Predicted Target")
    plt.plot([input_feature, input_feature], [actual_target, prediction], "k--")
    plt.annotate(f"Difference: {difference:.2f}", 
                 (input_feature, (actual_target + prediction) / 2), 
                 textcoords="offset points", xytext=(10, 10), ha='center')
    plt.title("Actual vs Predicted Target Value")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

def _index_of_closest(X: ArrayLike, k: float) -> int:
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx

if __name__ == '__main__':
    create_streamlit_app()
