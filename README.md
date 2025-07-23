Zomato Restaurant Clustering
Project Type
Unsupervised Clustering

Contribution
Individual

Project Summary
This project focuses on restaurant clustering using unsupervised machine learning techniques. The primary objective is to analyze a dataset of Zomato restaurants to uncover valuable insights that can inform business strategies, improve customer satisfaction, and optimize operational performance. The project tackles challenges such as identifying key factors that influence restaurant ratings, visualizing the geospatial distribution of restaurants, and processing a large, unstructured dataset to extract meaningful information.

The analysis utilizes two datasets: one with restaurant metadata (name, cost, cuisines, collections) and another with customer reviews. The process includes a comprehensive Exploratory Data Analysis (EDA), extensive data wrangling and feature engineering, and Natural Language Processing (NLP) to prepare the review text for modeling. The core of the project involves applying K-Means and Agglomerative Hierarchical Clustering algorithms to group the restaurants. Finally, the clusters are examined to provide actionable insights based on characteristics like price point, average rating, and cuisine types.

Problem Statement
In the rapidly growing food delivery and restaurant discovery industry, understanding customer preferences, restaurant performance, and geographical trends is vital for data-driven decision-making. Zomato, as a leading food-tech platform, accumulates massive datasets which, if analyzed effectively, can uncover hidden insights to improve business strategies, enhance customer satisfaction, and optimize operational performance.

However, the vastness and unstructured nature of this data pose challenges in terms of:

Identifying key factors influencing restaurant ratings and customer choices.

Visualizing geospatial distributions of restaurants and customer demand.

Cleaning, transforming, and analyzing high-volume data for meaningful insights.

This project aims to address these challenges by:

Performing exploratory data analysis (EDA) on Zomato’s dataset.

Deriving insights from restaurant ratings, cuisines, cost, and location.

Using data visualization tools to communicate trends effectively.

Providing actionable recommendations for restaurant businesses and platform improvements.

Libraries Used
pandas

numpy

matplotlib

seaborn

math

time

wordcloud

scipy

scikit-learn

nltk

contractions

gensim

shap

Data
The project uses two datasets:

Zomato Restaurant names and Metadata.csv: Contains information about restaurants like name, links, cost, collections, cuisines, and timings.

Zomato Restaurant reviews.csv: Contains customer reviews including the restaurant name, reviewer, review text, rating, metadata, time, and number of pictures.

Methodology
Data Loading and Initial Exploration: The datasets are loaded, and initial checks like .head(), .info(), .shape, and .describe() are performed.

Data Cleaning: Duplicate values are identified and handled. Missing values are checked and visualized using heatmaps.

Data Wrangling & Feature Engineering:

The 'Cost' column is converted to a numerical format.

'Cuisines' and 'Collections' are split to analyze individual items.

New features are created from the 'Metadata' and 'Time' columns in the reviews dataset.

Exploratory Data Analysis (EDA): In-depth analysis of variables is performed using various visualization techniques to understand distributions and relationships.

Text Preprocessing (NLP):

Contractions are expanded.

Text is converted to lowercase.

Punctuation, URLs, and digits are removed.

Stopwords and whitespace are removed.

Text is tokenized and lemmatized.

Sentiment Analysis: A sentiment score (positive/negative) is assigned to each review.

Feature Extraction: TF-IDF and Bag of Words techniques are used to convert text data into numerical vectors.

Dimensionality Reduction: Principal Component Analysis (PCA) is used to reduce the dimensionality of the feature space.

Clustering:

K-Means Clustering: The Elbow method and Silhouette score are used to find the optimal number of clusters. The model is then trained and clusters are assigned.

Agglomerative Hierarchical Clustering: A dendrogram is plotted to visualize the hierarchy and determine the number of clusters.

Topic Modeling: Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF) are used to identify topics within the reviews.

Cluster Analysis: The characteristics of the restaurants in each cluster are analyzed to derive business insights.

How to Run
Ensure you have Python and Jupyter Notebook installed.

Install the required libraries listed in the "Libraries Used" section. You can do this by running pip install <library_name> in your terminal. For some libraries like nltk, you may need to download additional data by running nltk.download() in a Python environment.

Place the two CSV files (Zomato Restaurant names and Metadata.csv and Zomato Restaurant reviews.csv) in the same directory as the Jupyter Notebook.

Open the Jupyter Notebook and run the cells sequentially.
