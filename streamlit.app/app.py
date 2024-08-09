import numpy as np
import streamlit as st
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Setting the title and description of the web app
st.title('Customer Behaviour of Electric Vehicle Adoption')
st.text('This is the web app to explore the customer behaviour for the adoption of electric vehicles in washington.')

# File uploader to allow users to upload their dataset
uploaded_file = st.file_uploader('Upload your file here')

if uploaded_file:
    st.header('Data Structure')
    df = pd.read_csv(uploaded_file)
    st.write(df.describe())

    st.header('Data Header')
    st.write(df.head())

    # Visualizations
    st.header('Visualizations')

    #Histogram of a selected numeric column
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if numeric_columns:
        selected_column = st.selectbox('Select a numeric column for histogram', numeric_columns)
        fig, ax = plt.subplots()
        ax.hist(df[selected_column], bins=20, edgecolor='k')
        ax.set_title(f'Histogram of {selected_column}')
        ax.set_xlabel(selected_column)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    #Correlation heatmap
    st.subheader('Correlation Heatmap')
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
    if not numeric_df.empty:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)
    else:
        st.text('No numeric columns available for correlation heatmap.')

    #Scatter plot of two selected numeric columns
    if len(numeric_columns) > 1:
        col1 = st.selectbox('Select x-axis column for scatter plot', numeric_columns, key='x')
        col2 = st.selectbox('Select y-axis column for scatter plot', numeric_columns, key='y')
        fig, ax = plt.subplots()
        ax.scatter(df[col1], df[col2])
        ax.set_title(f'Scatter Plot of {col1} vs {col2}')
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        st.pyplot(fig)

    #Box plot of a selected numeric column
    if numeric_columns:
        selected_box_column = st.selectbox('Select a numeric column for box plot', numeric_columns, key='box')
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_box_column], ax=ax)
        ax.set_title(f'Box Plot of {selected_box_column}')
        ax.set_xlabel(selected_box_column)
        st.pyplot(fig)

    #Line plot of a selected numeric column (if dataset contains time series data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        selected_line_column = st.selectbox('Select a numeric column for line plot', numeric_columns, key='line')
        fig, ax = plt.subplots()
        df.set_index('date')[selected_line_column].plot(ax=ax)
        ax.set_title(f'Line Plot of {selected_line_column} over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel(selected_line_column)
        st.pyplot(fig)




    # Models
    st.header('Models')

    # Linear Regression
    st.subheader('Linear Regression Model')
    if len(numeric_columns) > 1:
        target_column = st.selectbox('Select the target column for linear regression', numeric_columns)
        feature_columns = st.multiselect('Select the feature columns for linear regression', numeric_columns, default=numeric_columns)
        if st.button('Run Linear Regression'):
            data = df[feature_columns + [target_column]].dropna()
            X = data[feature_columns]
            y = data[target_column]
            if not X.empty and not y.empty:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                st.write(f'R-squared score: {score:.2f}')
                st.write('Model Coefficients:', model.coef_)

    # K-Means Clustering
    st.subheader('K-Means Clustering')
    if len(numeric_columns) > 1:
        cluster_columns = st.multiselect('Select columns for clustering', numeric_columns)
        num_clusters = st.slider('Select number of clusters', 2, 10, 3)
        if st.button('Run K-Means Clustering'):
            X = df[cluster_columns].dropna()
            if not X.empty:
                model = KMeans(n_clusters=num_clusters, random_state=42)
                clusters = model.fit_predict(X)
                df = df.loc[X.index]  # Align the DataFrame with the rows used in clustering
                df['cluster'] = clusters
                st.write('Cluster Centers:', model.cluster_centers_)
                fig, ax = plt.subplots()
                sns.scatterplot(x=X[cluster_columns[0]], y=X[cluster_columns[1]], hue=df['cluster'], palette='viridis', ax=ax)
                ax.set_title(f'K-Means Clustering with {num_clusters} Clusters')
                st.pyplot(fig)

    # Decision Tree Regression
    st.subheader('Decision Tree Regression Model')
    if len(numeric_columns) > 1:
        target_column_dt = st.selectbox('Select the target column for decision tree regression', numeric_columns, key='dt_target')
        feature_columns_dt = st.multiselect('Select the feature columns for decision tree regression', numeric_columns, default=numeric_columns, key='dt_features')
        if st.button('Run Decision Tree Regression'):
            data_dt = df[feature_columns_dt + [target_column_dt]].dropna()
            X_dt = data_dt[feature_columns_dt]
            y_dt = data_dt[target_column_dt]
            if not X_dt.empty and not y_dt.empty:
                X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_dt, y_dt, test_size=0.2, random_state=42)
                model_dt = DecisionTreeRegressor(random_state=42)
                model_dt.fit(X_train_dt, y_train_dt)
                score_dt = model_dt.score(X_test_dt, y_test_dt)
                st.write(f'R-squared score: {score_dt:.2f}')

    # Random Forest Classification
    st.subheader('Random Forest Classification Model')
    if len(categorical_columns) > 0 and len(numeric_columns) > 0:
        target_column_rf = st.selectbox('Select the target column for random forest classification', categorical_columns, key='rf_target')
        feature_columns_rf = st.multiselect('Select the feature columns for random forest classification', numeric_columns, default=numeric_columns, key='rf_features')
        if st.button('Run Random Forest Classification'):
            data_rf = df[feature_columns_rf + [target_column_rf]].dropna()
            X_rf = data_rf[feature_columns_rf]
            y_rf = data_rf[target_column_rf]
            if not X_rf.empty and not y_rf.empty:
                X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
                model_rf = RandomForestClassifier(random_state=42)
                model_rf.fit(X_train_rf, y_train_rf)
                score_rf = model_rf.score(X_test_rf, y_test_rf)
                st.write(f'Accuracy: {score_rf:.2f}')

