import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Data Science Portfolio", layout="wide")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Loading", "Preprocessing", "Model Selection", "Visualization"])

if page == "Home":
    st.title('My Data Science Portfolio')
    st.write("""
    Welcome to my Data Science Portfolio! This application demonstrates various data science skills using scikit-learn and Streamlit.

    Use the sidebar to navigate through different sections:
    - Data Loading: Load and explore datasets
    - Preprocessing: Apply data preprocessing techniques
    - Model Selection: Choose and train machine learning models
    - Visualization: Visualize results and insights
    """)

elif page == "Data Loading":
    st.title("Data Loading")

    # Predefined dataset buttons
    st.write("Load a predefined dataset:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load S&P 500 Data (Predict Change in Stock Price)"):
            st.session_state['file_path'] = "SP.csv"
            st.session_state['dataset_name'] = "Stock Price Prediction"
        if st.button("Load American Sign Language Alphabet Images (Warning Slow)"):
            st.session_state['file_path'] = "sign_mnist.csv"
            st.session_state['dataset_name'] = "Sign Language MNIST"
    with col2:
        if st.button("Load Text Emotional Classification"):
            st.session_state['file_path'] = "emotions.csv"
            st.session_state['dataset_name'] = "Emotion Prediction"
        if st.button("Load Country Happiness Data"):
            st.session_state['file_path'] = "country_happiness.csv"
            st.session_state['dataset_name'] = "Country Happiness"

    # Custom file uploader
    st.write("Or upload your own CSV file:")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Load the dataset
    if 'file_path' in st.session_state or uploaded_file is not None:
        try:
            if 'file_path' in st.session_state:
                df = pd.read_csv(st.session_state['file_path'])
            else:
                df = pd.read_csv(uploaded_file)
                st.session_state['dataset_name'] = "Custom Dataset"

            st.session_state['df'] = df

            # Display basic information about the dataset
            st.write(f"Dataset: {st.session_state['dataset_name']}")
            st.subheader("Sample data:")
            st.write(df.head())

            # Allow user to select multiple target columns
            target_columns = st.multiselect("Select target column(s)", df.columns.tolist())
            if target_columns:
                st.session_state['target_columns'] = target_columns
                st.session_state['feature_columns'] = [col for col in df.columns if col not in target_columns]
                #st.write(f"Target column(s): {target_columns}")
                #st.write(f"Feature columns: {st.session_state['feature_columns']}")

            # Add dataset-specific information
            if st.session_state['dataset_name'] == "Stock Price Prediction":
                st.write("This dataset is for predicting stock price changes (multi-output regression).")
            elif st.session_state['dataset_name'] == "Sign Language MNIST":
                st.write("This dataset is for predicting sign language gestures (multi-output classification).")
            elif st.session_state['dataset_name'] == "Emotion Prediction":
                st.write("This dataset is for predicting emotions based on text (NLP classification).")
            elif st.session_state['dataset_name'] == "Country Happiness":
                st.write("This dataset is for clustering countries based on happiness factors.")
                st.write("Note: The first column with country names should be ignored for clustering.")

            # Add a button to proceed to the next step
            if st.button("Proceed to Preprocessing"):
                st.session_state['page'] = "Preprocessing"
                st.experimental_rerun()

        except Exception as e:
            st.error(f"Error loading the dataset: {str(e)}")

    else:
        st.info("Please select a dataset or upload a CSV file to proceed.")

elif page == "Preprocessing":
    st.title("Data Preprocessing")

    if 'df' not in st.session_state or 'target_columns' not in st.session_state:
        st.warning("Please load a dataset and select target column(s) in the Data Loading page first.")
    else:
        df = st.session_state['df']
        target_columns = st.session_state['target_columns']
        feature_columns = st.session_state['feature_columns']

        if st.button("Apply Preprocessing"):
            preprocessed_data = df[feature_columns].copy()

            # Step 1: Vectorize text columns
            text_columns = preprocessed_data.select_dtypes(include=['object']).columns
            if len(text_columns) > 0:
                st.write("Vectorizing text columns")
                for col in text_columns:
                    #st.write(f"- {col}")
                    vectorizer = TfidfVectorizer(max_features=100)
                    text_features = vectorizer.fit_transform(preprocessed_data[col].fillna(''))
                    text_feature_names = [f"{col}_{i}" for i in range(text_features.shape[1])]
                    preprocessed_data = pd.concat([preprocessed_data.drop(col, axis=1),
                                                   pd.DataFrame(text_features.toarray(), columns=text_feature_names)],
                                                  axis=1)
            else:
                st.write("No text columns to vectorize.")

            # Step 2: Standardize numerical columns
            numeric_columns = preprocessed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                st.write("Standardizing numerical columns")
                scaler = StandardScaler()
                preprocessed_data[numeric_columns] = scaler.fit_transform(preprocessed_data[numeric_columns])
            else:
                st.write("No numerical columns to standardize.")

            st.session_state['preprocessed_features'] = preprocessed_data
            st.session_state['target'] = df[target_columns]

            st.write("Preprocessing complete.")
            st.write("Preprocessed data sample:")
            st.write(preprocessed_data.head())

            # Display shape of preprocessed data
            #st.write(f"Shape of preprocessed features: {preprocessed_data.shape}")
            #st.write(f"Shape of target data: {st.session_state['target'].shape}")

        else:
            st.write("Click 'Apply Preprocessing' to preprocess the data.")

elif page == "Model Selection":
    st.title("Model Selection")

    if 'preprocessed_features' not in st.session_state:
        st.warning("Please preprocess the data in the Preprocessing page first.")
    else:
        preprocessed_features = st.session_state['preprocessed_features']
        target = st.session_state['target']

        # Model selection
        model_type = st.selectbox("Select problem type",
                                  ["Multiclass classification", "Multilabel classification",
                                   "Multiclass multioutput classification", "Multioutput regression"])

        # Model options (you can expand this list based on the README requirements)
        model_options = {
            "Linear Models": None,  # Add appropriate models here
            "SVM": None,
            "Stochastic Gradient Descent": None,
            "K Neighbors": None,
            "Decision Trees": None,
            "Random Forests": None,
            "Multi-layer Perceptron": None
        }

        selected_model = st.selectbox("Select a model", list(model_options.keys()))

        st.write("Selected model:", selected_model)
        st.write("This section will be expanded to include model training and evaluation.")

elif page == "Visualization":
    st.title("Data Visualization")

    if 'preprocessed_features' not in st.session_state or 'target' not in st.session_state:
        st.warning("Please preprocess the data in the Preprocessing page first.")
    else:
        preprocessed_features = st.session_state['preprocessed_features']
        target = st.session_state['target']

        # Visualization options
        viz_type = st.selectbox("Select visualization type", ["Clustering", "Dimensionality Reduction"])

        if viz_type == "Clustering":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(preprocessed_features)

            # Visualize using ICA
            ica = FastICA(n_components=2, random_state=42)
            ica_results = ica.fit_transform(preprocessed_features)

            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(ica_results[:, 0], ica_results[:, 1], c=cluster_labels, cmap='viridis')
            plt.colorbar(scatter)
            plt.title(f"ICA visualization of {n_clusters} clusters")

            # Add target labels
            if isinstance(target, pd.DataFrame) and target.shape[1] == 1:
                target_values = target.iloc[:, 0]
                for i, txt in enumerate(target_values):
                    ax.annotate(txt, (ica_results[i, 0], ica_results[i, 1]), xytext=(5, 5), textcoords='offset points',
                                fontsize=5, alpha=0.7)
            elif isinstance(target, pd.Series):
                for i, txt in enumerate(target):
                    ax.annotate(txt, (ica_results[i, 0], ica_results[i, 1]), xytext=(5, 5), textcoords='offset points',
                                fontsize=5, alpha=0.7)

            st.pyplot(fig)

        elif viz_type == "Dimensionality Reduction":
            dim_reduction_method = st.selectbox("Select dimensionality reduction method",
                                                ["ICA", "PCA", "Kernel PCA", "t-SNE"])
            n_components = st.slider("Number of components", 2, 10, 2)

            if dim_reduction_method == "ICA":
                ica = FastICA(n_components=n_components)
                reduced_data = ica.fit_transform(preprocessed_features)
            elif dim_reduction_method == "PCA":
                pca = PCA(n_components=n_components)
                reduced_data = pca.fit_transform(preprocessed_features)
            elif dim_reduction_method == "Kernel PCA":
                kpca = KernelPCA(n_components=n_components, kernel='rbf')
                reduced_data = kpca.fit_transform(preprocessed_features)
            else:  # t-SNE
                tsne = TSNE(n_components=n_components, random_state=42)
                reduced_data = tsne.fit_transform(preprocessed_features)

            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1])
            plt.title(f"{dim_reduction_method} visualization with {n_components} components")

            # Add target labels
            if isinstance(target, pd.DataFrame) and target.shape[1] == 1:
                target_values = target.iloc[:, 0]
                for i, txt in enumerate(target_values):
                    ax.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]), xytext=(5, 5),
                                textcoords='offset points', fontsize=8, alpha=0.7)
            elif isinstance(target, pd.Series):
                for i, txt in enumerate(target):
                    ax.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]), xytext=(5, 5),
                                textcoords='offset points', fontsize=8, alpha=0.7)

            st.pyplot(fig)

        # Add color coding for multiple targets
        if isinstance(target, pd.DataFrame) and target.shape[1] > 1:
            st.write("Multiple target columns detected. Showing color-coded plot for the first target column.")

            # Use the first target column for color coding
            first_target = target.iloc[:, 0]
            unique_targets = first_target.unique()
            color_map = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_targets)))

            fig, ax = plt.subplots(figsize=(10, 6))
            for i, target_value in enumerate(unique_targets):
                mask = first_target == target_value
                ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], c=[color_map[i]], label=target_value,
                           alpha=0.7)

            ax.legend()
            plt.title(
                f"{dim_reduction_method} visualization with {n_components} components (color-coded by {target.columns[0]})")
            st.pyplot(fig)

# Run the Streamlit app
if __name__ == '__main__':
    st.sidebar.info("Select a page above to get started.")