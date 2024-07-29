import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import io
import os

# Load dataset
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'breast-cancer.csv')
df = pd.read_csv(file_path)

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



columns = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
       'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
       'smoothness_se', 'compactness_se', 'concave points_se', 'symmetry_se',
       'symmetry_worst']
sample_data = pd.DataFrame(columns=columns)
# Streamlit app
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('Confusion Matrix')
    st.pyplot(fig)

# Create a downloadable template file
def create_template_file():
    columns = list(x_corr.columns.difference(high_corr_columns))
    sample_data = pd.DataFrame(columns=columns)
    towrite = io.BytesIO()
    sample_data.to_csv(towrite, index=False)
    towrite.seek(0)
    return towrite

# Provide download link for the template
def provide_template_download_link():
    st.header("Parametric Cancer Predictive Analytics Platform- Breast Cancer// Divyansh Sankhla")
    template_file = create_template_file()
    st.download_button(
        label="Download Template CSV",
        data=template_file,
        file_name='cancer_prediction_template.csv',
        mime='text/csv'
    )
file_path2 = os.path.join(script_dir, 'images/breast_cancer.jpg')
file_path3 = os.path.join(script_dir, 'images/breast_cancer2.webp')
file_path4 = os.path.join(script_dir, 'images/OIP.jpg')
images = [
    file_path2,file_path3,file_path4
]
st.image(images, width=200)

def create_download_link(df, filename):
    towrite = io.BytesIO()
    if filename.endswith('.xlsx'):
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
    elif filename.endswith('.csv'):
        df.to_csv(towrite, index=False)
    towrite.seek(0)
    st.download_button(
        label="Download Results",
        data=towrite,
        file_name=filename,
        mime="application/vnd.ms-excel" if filename.endswith('.xlsx') else "text/csv"
    )
provide_template_download_link()
def classification_reporting(y_test, y_pred):
    cr = classification_report(y_test, y_pred)
    st.text(cr)

# File uploader
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Load the uploaded file
    try:
        if uploaded_file.name.endswith('.csv'):
            uploaded_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            uploaded_df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Check columns and preprocess data
        required_columns = x_corr.columns.difference(high_corr_columns)


        if all(col in uploaded_df.columns for col in required_columns):
            # Process the uploaded data
            input_data = uploaded_df[required_columns].values
            
            # Scale the input data
            input_data = ss.fit_transform(input_data)
            
            # Make predictions
            predictions = lg.predict(input_data)
            
            # Prepare results DataFrame
            uploaded_df['Prediction'] = ['Having Cancer' if pred == 1 else 'Chill, you are fit' for pred in predictions]
            
            # Show results in table
            st.write("Prediction Results:")
            st.dataframe(uploaded_df)
            # Provide download link for results
            create_download_link(uploaded_df, 'prediction_results.xlsx')
            # create_download_link(uploaded_df, 'prediction_results.csv')
        else:
            st.error("Uploaded file does not contain the correct columns. Please check the format.")
    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    # Create input fields for the retained features
    st.write("OR")
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
            st.write("Information about the model")
            st.write("This is the Confusion matrix of the Model on which this Algo is Trained")
            plot_confusion_matrix(y_test, y_pred)
            st.write("This is the Classification report of the Model on which this Algo is Trained")
            classification_reporting(y_test, y_pred)
        except ValueError:
            st.error("Please ensure all inputs are filled correctly and are numerical.")
