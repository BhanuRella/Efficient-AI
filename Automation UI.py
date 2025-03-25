import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance

def load_data():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def eda(df):
    st.write("# Exploratory Data Analysis (EDA)")
    
    st.write("#### Statistical Summary")
    st.write(df.describe())
    
    st.write("#### Data Visualization")
    selected_columns = st.multiselect("Select Columns to Plot", df.columns)
    plot_type = st.selectbox("Select Plot Type", ["Histogram", "Boxplot", "Scatter Plot"])
    
    if selected_columns:
        fig, ax = plt.subplots()
        if plot_type == "Histogram":
            for col in selected_columns:
                sns.histplot(df[col], kde=True, ax=ax)
        elif plot_type == "Boxplot":
            sns.boxplot(data=df[selected_columns], ax=ax)
        elif plot_type == "Scatter Plot" and len(selected_columns) == 2:
            sns.scatterplot(x=df[selected_columns[0]], y=df[selected_columns[1]], ax=ax)
        st.pyplot(fig)

def preprocess_data(df):
    st.write("# Preprocessing the Data")
    if st.checkbox("Handle Missing Values"):  
        df = df.fillna(method='ffill')
        st.write("Missing values filled with previous valid data point")
    
    if st.checkbox("Standardize Numeric Data"):  
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
        st.write("Numeric data standardized")
    
    if st.checkbox("Encode Categorical Data"):  
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
        st.write("Categorical data encoded")
    
    return df

def feature_selection(df, target):
    st.write("# Feature Selection")
    st.write('### -Must Preprocess the data to continue -')
    available_features = [col for col in df.columns if col != target]
    
    # Train a model to get feature importances
    X = df[available_features]
    y = df[target]
    model = RandomForestRegressor() if y.dtype in ['int64', 'float64'] else RandomForestClassifier()
    model.fit(X, y)
    
    # Compute feature importances
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({"Feature": available_features, "Importance": importance}).sort_values(by="Importance", ascending=False)
    st.write("#### Feature Importance Table")
    st.dataframe(feature_importance_df)
    
    selected_features = st.multiselect("Select Features for Model Training", available_features, default=feature_importance_df["Feature"].tolist()[:5])
    st.write(f"Selected Features: {selected_features}")
    return df[selected_features + [target]]

def train_model(df, target):
    st.write("### Model Training & Evaluation")
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if y.dtype in ['int64', 'float64']:  
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        st.write(f"Mean Squared Error: {mse}")
    else:  
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Accuracy Score: {accuracy}")

def main():
    st.title("Automated Data Pipeline Optimization for ML Inference")
    df = load_data()
    if df is not None:
        st.write("### Data Preview")
        st.dataframe(df.head())
        eda(df)
        target = st.selectbox("Select Target Column", df.columns)
        df = preprocess_data(df)
        df = feature_selection(df, target)
        train_model(df, target)

if __name__ == "__main__":
    main()
