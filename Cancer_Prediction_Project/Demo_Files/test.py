import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('breast-cancer.csv')

# Map diagnosis to binary
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Split into features and target
x = df.drop(['diagnosis', 'id'], axis=1)
y = df['diagnosis']

# Correlation matrix
x_corr = x.corr()

# Function to find high correlation columns
def correlation(dataset, threshold):
    s = set()
    for i in range(len(dataset.columns)):
        for j in range(i):
            if abs(dataset.iloc[i, j]) > threshold:
                colname_i = dataset.columns[i]
                colname_j = dataset.columns[j]
                s.add(colname_j)
    return s

# Drop high correlation columns
high_corr_columns = correlation(x_corr, 0.8)
x = x.drop(high_corr_columns, axis=1)

# Feature Scaling
ss = StandardScaler()
x = ss.fit_transform(x)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

# Train the Logistic Regression model
lg = LogisticRegression()
lg.fit(x_train, y_train)
y_pred = lg.predict(x_test)

# Prediction function
def prediction_model(input_df):
    input_df = np.asarray(input_df, dtype=np.float64)
    pred = lg.predict(input_df.reshape(1, -1))
    if pred == 1:
        return "Having Cancer"
    else:
        return "Chill, you are fit"

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('Confusion Matrix')
    
    st.pyplot(fig)

def classification_reporting(y_test, y_pred):
    cr = classification_report(y_test, y_pred)
    st.text(cr)



# Streamlit app
st.set_page_config(page_title="Cancer Prediction ML Model", layout="centered", initial_sidebar_state="collapsed")
st.header("Cancer Prediction ML Model")
images = [
    'breast_cancer.jpg',
    'breast_cancer2.webp','OIP.jpg'
]
st.image(images, width=200)
st.balloons()
# Create input fields for the retained features
input_data = []
for feature in x_corr.columns:
    if feature not in high_corr_columns:
        value = st.text_input(f"Enter value for {feature}", key=feature)
        input_data.append(value)

submit = st.button("Generate")

if submit:
    try:
        input_data = [float(i) for i in input_data]
        result = prediction_model(input_data)
        st.write(result)
        st.write("This is the Confusion matrix of the Model on which this Algo is Trained")
        plot_confusion_matrix(y_test, y_pred)
        st.write("This is the Calssification report of the Model on which this Algo is Trained")
        classification_reporting(y_test, y_pred)
    except ValueError:
        st.error("Please ensure all inputs are filled correctly and are numerical.")
