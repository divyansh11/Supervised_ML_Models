Problem Statement
The goal of this project is to build a machine learning model that can predict whether a patient has breast cancer based on features extracted from a medical dataset. The model uses a logistic regression classifier to provide predictions and insights into the likelihood of cancer diagnosis.

Project Details
Dataset Source: Breast Cancer Wisconsin (Diagnostic) Data Set
Features: The dataset contains various features related to breast cancer diagnosis, including measurements of cell nuclei, texture, perimeter, area, and more.
Target Variable: diagnosis (Malignant or Benign)

Features Developed
Model Training: Trained a Logistic Regression model on the breast cancer dataset.Used feature scaling to normalize the data and improve model performance.
Feature Selection:Implemented a correlation-based feature selection to drop highly correlated features, improving model efficiency and reducing redundancy.
Confusion Matrix: Visualized the confusion matrix to assess the performance of the model, including metrics such as precision, recall, and F1-score.

Streamlit Web Application:
Input Handling: Users can input feature values directly or upload a CSV/Excel file with the required data.
Model Prediction: The application predicts whether the input data indicates cancer or not.
Results Display: Predictions are displayed in a table, and users can download the results as a CSV or Excel file.
Downloadable Template: Provides a downloadable template file for users to fill out and upload, ensuring data is in the correct format.

Features
Data Upload: Upload a CSV or Excel file with patient data to get predictions.
Interactive Input: Enter individual feature values through text inputs for quick predictions.
Downloadable Results: Get the prediction results in a downloadable format.
Confusion Matrix Visualization: View a graphical representation of the model’s performance.


![image](https://github.com/user-attachments/assets/322bbe44-d2e5-49db-b134-85f7df878c69)


Requirements
Python 3.x
Streamlit
Pandas
NumPy
scikit-learn
Matplotlib
openpyxl (for Excel file handling)

Example Usage
Download Template: Click the "Download Template CSV" button to get a sample file with the required columns.
Upload File: Upload your filled-out CSV or Excel file.
View Predictions: The application will display predictions and allow you to download the results.
