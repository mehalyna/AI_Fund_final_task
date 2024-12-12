# Regression Model Training and Streamlit Prediction App

This project demonstrates the end-to-end workflow of training a linear regression model, saving the trained model and datasets, and deploying a Streamlit web application for predictions and visualizations. The goal is to provide an example of a basic machine learning pipeline using Python and popular libraries.

---

## **Purpose**
The project is divided into two main parts:
1. **Model Training**: 
   - Train a linear regression model using synthetic regression data.
   - Save the trained model and datasets for future use.
   - Evaluate the model's performance with metrics like Mean Squared Error (MSE).

2. **Streamlit App**:
   - Provide a user-friendly interface for making predictions using the saved regression model.
   - Visualize the relationship between input features, actual target values, and predictions.

---

## **Features**
- Train and evaluate a linear regression model using `scikit-learn`.
- Save the trained model and datasets using `joblib`.
- Interactive prediction using a Streamlit app.
- Visualization of differences between actual and predicted target values.

---

## **Requirements**
The following Python libraries are required:
- `numpy`
- `scikit-learn`
- `joblib`
- `streamlit`
- `matplotlib`

Install all dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## **Instructions**

### **1. Clone the Repository**
```bash
git clone mehalyna/AI_Fund_final_task
cd AI_Fund_final_task
```

### **2. Train the Model**
Run the `train_and_save_model.py` script to train the regression model, evaluate it, and save the necessary files:
```bash
python train_and_save_model.py
```

This script will:
- Train a linear regression model.
- Save the model as `linear_regression_model.joblib`.
- Save the datasets (`X.joblib` and `y.joblib`).

### **3. Start the Streamlit App**
Run the Streamlit app for making predictions and visualizing results:
```bash
streamlit run streamlit_app.py
```

### **4. Use the App**
- Open the URL displayed in the terminal (usually `http://localhost:8501`).
- Adjust the slider to select a feature value between `-3.0` and `3.0`.
- Click the "Predict value" button to:
  - See the predicted target value.
  - Visualize the difference between the actual and predicted values.

---

## **File Structure**
```
├── train_and_save_model.py  # Script for training the regression model and saving datasets.
├── streamlit_app.py         # Streamlit application for prediction and visualization.
├── requirements.txt         # Required libraries for the project.
├── README.md                # Project documentation.
├── linear_regression_model.joblib  # Saved regression model.
├── X.joblib                 # Saved feature dataset.
├── y.joblib                 # Saved target dataset.
```

---

## **Example Usage**
1. **Training Output Example**:
   ```
   Model trained and saved as linear_regression_model.joblib.
   Datasets saved as X.joblib and y.joblib.
   Mean Squared Error: 416.80
   ```

2. **Streamlit App Interface**:
   - Input slider to select feature values.
   - Button to predict and visualize results.
   - Scatter plot showing dataset, actual, and predicted values.

---

## **Notes**
- Ensure that `linear_regression_model.joblib`, `X.joblib`, and `y.joblib` are in the same directory as the Streamlit app script for proper functionality.
- Modify the slider range or visualization options in `streamlit_app.py` to suit your dataset or use case.

---

## **License**
This project is licensed under the MIT License. Feel free to modify and use it for educational purposes.
