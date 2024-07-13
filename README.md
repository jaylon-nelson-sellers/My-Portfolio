# My-Portfolio
My current portfolio demonstrating my data science skills. 

Keep seperate steps on seperate pages
Write concise explanations of the processes going on.
# Feature List
- Load dataset, allow user to select column(s)
- - Preprocess data automatically 
  - Standardize
  - Vectorize Text
- Datasets
  - 1: sp: Predict the change in stock price (multi output regression)
  - 2: sign_mnist: Predict the sign language (multi output classification)
  - 3: emotions: Predict the emotion based on text (NLP classification)
  - 4: country_happiness: cluster the countries (ingmore the first column with country names)
- Set functions for different types of problems
  - Multiclass classification
  - Multilabel classification
  - Multiclass multioutput classification
  - multioutput regression
- Implement "Pipelines" for program: code should be loosely coupled and highly cohesive
- Allow users to select from the following models
  - Linear Models ()
  - SVM
  - Stochastic Gradient Descent
  - K Neighbors
  - Decision Trees
  - (Histo)Gradient Boosted Trees
  - Random Forests
  - Multi layer perceptron
  - pick 3 Clustering models
- Data transformation
  - Allow users to create a pipeline of data transformation (transformer -> model)
  - Decomposition: Allow user to set the number of components with the following models and augment the dataset 
  - (save as a copy of dataset)
    - models:
      - ICA
      - KernalPCA
      - TSNE
  - Pre clustering: 
    - (allow users to pick cluster number and Cluster the data, append cluster to dataset and feed to model 
- Visualize Data throughout process
  - Visualize the clustering process via the elbow method 
  - Clustering: Visualize Data with TSNE (2 components), allow user to pick the number of clusters and output this to screen (high dpi)
  - Regression: Show results: MSE, R2
  - Classification: Accuracy, Confusion Matrix