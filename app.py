import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression , LinearRegression
from sklearn.svm import SVC , SVR

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load data
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
        return df
    return None

def main():
    # Set page title and background
    st.set_page_config(page_title="Streamlit App with EDA and Model Evaluation", layout="wide", page_icon=":bar_chart:")

    # Custom HTML header
    html_temp = """
    <div style="background-color:#ffd700;padding:10px;border-radius:10px">
    <h1 style="color:Black;text-align:center;">Streamlit App with EDA and Model Evaluation</h1>
    </div>
    """
    html_temp2 = """
    <h2 align="center"><span style="color:#ffd700;">Exploratiary Data Analysis (EDA)</span></h2>
    """
    html_temp3 = """
    <h2 align="center"><span style="color:#ffd700;">Upload your data</span></h2>
    """
    html_temp4 = """
    <h2 align="center"><span style="color:#ffd700;">Handle Missing Values</span></h2>
    """
    html_temp5 = """
    <h2 align="center"><span style="color:#ffd700;">Handle Duplicates</span></h2>
    """
    html_temp6 = """
    <div style="background-color:#ffd700;padding:10px;border-radius:10px">
    <h1 style="color:Black;text-align:center;">Model Evaluation</h1>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    # Load data
    st.markdown(html_temp3, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your data", type=["csv", "txt"])
    df = load_data(uploaded_file)

    # Check if data is uploaded
    if df is not None:
        # EDA
        st.markdown(html_temp2, unsafe_allow_html=True)

        # Data preview
        with st.expander("Show Data Preview"):
            st.write(df.head(10))

        # Data summary
        with st.expander("Show Data Summary"):
            st.write(df.describe())

        # Data types
        with st.expander("Show Data Types"):
            st.write(df.dtypes)

        # Missing values
        st.markdown(html_temp4, unsafe_allow_html=True)
        with st.expander("Show Missing Values"):
            missing_values = df.isnull().sum()
            st.write(missing_values)

        # Handle missing values only if they exist
        if missing_values.any():
            with st.expander("Handle Missing Values"):
                missing_cols = st.multiselect("Select columns with missing values", df.columns)
                if missing_cols:
                    cat_cols = df[missing_cols].select_dtypes(include=['object']).columns
                    num_cols = df[missing_cols].select_dtypes(exclude=['object']).columns
                    for col in cat_cols:
                        if df[col].isnull().sum() > 0:
                            imputer_cat = SimpleImputer(strategy='most_frequent')
                            df[col] = imputer_cat.fit_transform(df[[col]]).ravel()
                    for col in num_cols:
                        if df[col].isnull().sum() > 0:
                            imputer_num = SimpleImputer(strategy='mean')
                            df[col] = imputer_num.fit_transform(df[[col]]).ravel()
                    # Update missing values after handling
                    missing_values = df.isnull().sum()

        # Update missing values display
        with st.expander("Updated Missing values:"):
            st.write(missing_values)

        st.markdown(html_temp5, unsafe_allow_html=True)
        # Duplicates values
        with st.expander("Show Duplicate Values"):
            duplicate_values = df.duplicated().sum()
            st.write(duplicate_values)

        # Handle duplicates values
        with st.expander("Handle Duplicates Values"):
            df.drop_duplicates(inplace=True)
            # Update duplicate values count after removing duplicates
            duplicate_values = df.duplicated().sum()
            st.write("Duplicates values removed successfully!")

        # Update duplicate values display
        with st.expander("Updated Duplicate Values:"):
            st.write(duplicate_values)

        # Model Building
        st.markdown(html_temp6, unsafe_allow_html=True)

        html_prob = """
        <h2 align="center"><span style="color:#ffd700;">Select Problem</span></h2>
        """
        html_prob1 = """
        <h2 align="center"><span style="color:#ffd700;">Select Model</span></h2>
        """
        html_prob3 = """
        <h2 align="center"><span style="color:#ffd700;">Select Target Variable</span></h2>
        """
        html_prob2 = """
        <h2 align="center"><span style="color:#ffd700;">Evaluate Results</span></h2>
        """
        st.markdown(html_prob, unsafe_allow_html=True)
        # Problem type selection
        # Problem type selection
        problem_type = st.radio("Select a problem type", (None, "Classification", "Regression"))
        model_name = None  # Initialize model_name here
        if problem_type==None:
            st.write("Please select a problem type")
        else:
        # Model dictionary for both classification and regression
            models = {
                "Classification": {
                    "Random Forest Classifier": RandomForestClassifier(),
                    "Decision Tree Classifier": DecisionTreeClassifier(),
                    "Logistic Regression Classifier": LogisticRegression(),
                    "Support Vector Machine Classifier": SVC()
                },
                "Regression": {
                    "Random Forest Regressor": RandomForestRegressor(),
                    "Decision Tree Regressor": DecisionTreeRegressor(),
                    "Linear Regression Regressor": LinearRegression(),
                    "Support Vector Machine Regressor": SVR()
                }
            }

            # Select model based on problem type
            st.markdown(html_prob1, unsafe_allow_html=True)
            model_options = models[problem_type].keys()
            model_name = st.selectbox("", [""] + list(model_options))

        if model_name:

            # Target Variable selection]
            st.markdown(html_prob3, unsafe_allow_html=True)
            
            target_variable = st.selectbox("Select Target Variable", df.columns)

            X = df.drop([target_variable], axis=1)
            y = df[target_variable]

            # Handle missing values
            if X.isnull().sum().any():
                st.warning("There are missing values in the dataset. Please handle them before model building.")

            # Data preprocessing
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            label_encoder = LabelEncoder()
            scaler = StandardScaler()
            for col in X.columns:
                if X[col].dtype == "object":
                    X[col] = label_encoder.fit_transform(X[col])
            X = scaler.fit_transform(X)

            # Train Test Split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Model building
            model = models[problem_type][model_name]
            model.fit(X_train, y_train)

            # Prediction
            y_pred = model.predict(X_test)

            # Evaluation
            st.markdown(html_prob2, unsafe_allow_html=True)
            with st.expander("Show Evaluation Metrics"):

                if problem_type == "Classification":
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='macro')
                    recall = recall_score(y_test, y_pred, average='macro')
                    f1 = f1_score(y_test, y_pred, average='macro')
                    st.write(f"Accuracy: {accuracy}")
                    st.write(f"Precision: {precision}")
                    st.write(f"Recall: {recall}")
                    st.write(f"F1 Score: {f1}")
                elif problem_type == "Regression":
                    from sklearn.metrics import r2_score, mean_squared_error
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    st.write(f"R2 Score: {r2}")
                    st.write(f"Mean Squared Error: {mse}")


if __name__ == "__main__":
    main()
