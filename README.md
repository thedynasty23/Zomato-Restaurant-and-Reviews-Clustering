Zomato Restaurant and Reviews Clustering
A comprehensive machine learning project analyzing Zomato restaurant data and customer reviews through advanced clustering techniques to derive actionable business insights for the food delivery industry in Hyderabad's Gachibowli area.

🚀 Project Overview
This project performs in-depth analysis of 105 restaurants and 10,000 customer reviews from Zomato's platform using unsupervised machine learning algorithms. The analysis identifies distinct restaurant categories, customer review patterns, and dining preferences to help stakeholders make informed decisions.

✨ Key Features & Results
🍽️ Restaurant Analysis
Price Range Analysis: ₹150 - ₹2,800 (Most common: ₹500 - 13 restaurants)

Top 5 Expensive Restaurants:

Collage - Hyatt Hyderabad Gachibowli: ₹2,800

Feast - Sheraton Hyderabad Hotel: ₹2,500

Jonathan's Kitchen: ₹1,900

10 Downing Street: ₹1,900

Cascade - Radisson Hyderabad Hitec City: ₹1,800

Top 5 Budget Restaurants:

Amul: ₹150

Mohammedia Shawarma: ₹150

Sweet Basket: ₹200

Hunger Maggi Point: ₹200

Momos Delight: ₹200

📊 Reviews & Ratings Analysis
Total Reviews Analyzed: 9,964 (after data cleaning)

Rating Distribution: Most restaurants rated 4-5 stars

Peak Review Period: 2018 (4,903 reviews), 2019 (4,802 reviews)

Top 5 Highest Rated Restaurants:

AB's - Absolute Barbecues: 4.88 ⭐ (₹1,500 price point)

B-Dubs: 4.81 ⭐ (₹1,600 price point)

3B's - Buddies, Bar & Barbecue: 4.76 ⭐ (₹1,100 price point)

Paradise: 4.70 ⭐ (₹800 price point)

Flechazo: 4.66 ⭐ (₹1,300 price point)

Lowest Rated Restaurants:

Hotel Zara Hi-Fi: 2.40 ⭐ (₹400 price point)

Asian Meal Box: 2.58 ⭐ (₹200 price point)

Pakwaan Grand: 2.71 ⭐ (₹400 price point)

Mathura Vilas: 2.82 ⭐ (₹500 price point)

Behrouz Biryani: 2.83 ⭐ (₹600 price point)

🥘 Cuisine Popularity
Top 5 Most Popular Cuisines:

North Indian: 61 restaurants

Chinese: 43 restaurants

Continental: 21 restaurants

Biryani: 16 restaurants

Fast Food: 15 restaurants

🏷️ Collection Tags Analysis
Most Used Restaurant Tags:

Great Buffets: 11 restaurants

Food Hygiene Rated Restaurants in Hyderabad: 8 restaurants

Live Sports Screenings: 7 restaurants

Hyderabad's Hottest: 7 restaurants

Corporate Favorites: 6 restaurants

👥 Reviewer Insights
Top Food Critics (by followers):

Satwinder Singh: 13,410 followers, 186 reviews, 3.67 avg rating

Eat_vth_me: 13,320 followers, 60 reviews, 5.00 avg rating

Samar Sardar: 11,329 followers, 8 reviews, 3.50 avg rating

Foodies Hyderabad: 9,494 followers, 31 reviews, 4.50 avg rating

Srinivas: 7,628 followers, 34 reviews, 3.71 avg rating

🛠️ Technology Stack & Implementation
Core Technologies
python
# Data Analysis & Processing
pandas==2.2.2
numpy==1.26.4
scipy==1.13.1

# Machine Learning
scikit-learn==1.6.1
statsmodels==0.14.5

# Visualization
matplotlib
seaborn
plotly
wordcloud

# Natural Language Processing
nltk
textblob

# Development Environment
jupyter-notebook
python==3.11+
Machine Learning Pipeline
python
# Feature Engineering Process
1. Data Type Conversion: Cost (string → int64), Rating (object → float)
2. Feature Extraction: Reviewer_Total_Review, Reviewer_Followers from Metadata
3. Time Features: Review_Year, Review_Month, Review_Hour
4. Text Processing: Review text normalization, tokenization, lemmatization
5. Encoding: TF-IDF and Bag of Words for review text
📁 Project Structure
text
Zomato-Restaurant-and-Reviews-Clustering/
├── README.md                                    # Project documentation
├── Zomato_Restaurant_Clustering.ipynb          # Main analysis notebook
├── Dataset/                                     # Data directory
├── Zomato Restaurant names and Metadata.csv    # Restaurant metadata (105 rows)
└── Zomato Restaurant reviews.csv               # Customer reviews (10,000 rows)
📈 Machine Learning Results
K-Means Clustering Performance
python
# Optimal Parameters
n_clusters = 5
algorithm = 'K-Means'
silhouette_score = 0.247
davies_bouldin_score = 1.151

# Cluster Distribution
Cluster 0: 22 restaurants (Budget-friendly, diverse cuisines)
Cluster 1: 25 restaurants (Mid-range casual dining)
Cluster 2: 20 restaurants (Premium dining, international cuisines)
Cluster 3: 17 restaurants (Fast food and street food)
Cluster 4: 12 restaurants (High-end luxury dining)
Agglomerative Hierarchical Clustering
python
# Performance Metrics
silhouette_score = 0.247
davies_bouldin_score = 1.151
linkage_method = 'ward'
n_clusters = 5
Cluster Characteristics
Cluster 0 - Budget Diverse (22 restaurants)

Cuisines: Biryani, North Indian, Chinese, Seafood, Healthy Food, Fast Food

Price Range: ₹150-₹800

Examples: Amul, Hotel Zara Hi-Fi, Behrouz Biryani

Cluster 1 - Mid-Range Casual (25 restaurants)

Cuisines: Continental, American, Chinese, North Indian, Arabian

Price Range: ₹500-₹1,200

Examples: The Foodie Monster Kitchen, La La Land, Deli 9 Bistro

Cluster 2 - Premium International (20 restaurants)

Cuisines: Chinese, Continental, European, Mediterranean, Seafood

Price Range: ₹1,200-₹2,000

Examples: Beyond Flavours, The Fisherman's Wharf, Over The Moon Brew Company

Cluster 3 - Quick Service (17 restaurants)

Cuisines: Lebanese, Ice Cream, Street Food, Fast Food, Pizza

Price Range: ₹150-₹600

Examples: Shah Ghouse Spl Shawarma, KFC, Domino's Pizza

Cluster 4 - Luxury Fine Dining (12 restaurants)

Cuisines: Biryani, North Indian, Asian, Mediterranean, Modern Indian

Price Range: ₹1,500-₹2,800

Examples: AB's - Absolute Barbecues, Collage - Hyatt, Feast - Sheraton

📊 Statistical Analysis Results
Hypothesis Testing Results
python
# Test 1: Cost vs Rating Correlation
H0: No relationship between cost and rating (β1 = 0)
H1: Positive relationship between cost and rating (β1 > 0)
Method: Linear Regression
Result: Reject H0 (p < 0.05)
Conclusion: Significant relationship exists

# Test 2: Reviewer Followers vs Rating
H0: Reviewer followers don't affect rating (β1 = 0)  
H1: More followers lead to higher ratings (β1 > 0)
Method: Linear Regression
Result: Reject H0 (p < 0.05)

# Test 3: Cuisine Variety vs Rating
H0: Cuisine variety doesn't affect rating (β3 = 0)
H1: More variety leads to higher ratings (β3 > 0)
Method: Chi-squared Test
Result: Reject H0 (p < 0.05)
Data Quality Metrics
python
# Missing Value Treatment
Restaurant Dataset:
- Collections: 51.43% missing → Column dropped
- Timings: 1 missing value → Mode imputation

Reviews Dataset:
- Duplicates: 36 rows → Removed
- Reviewer_Followers: 1,578 missing → Filled with 0
- Review text: 7 missing → Filled with "No Review"

# Final Dataset Size
Merged Dataset: 9,961 rows × 17 columns
Restaurants Analyzed: 101 (4 without reviews)
🚀 Getting Started
Prerequisites
bash
python >= 3.11
jupyter-notebook
git
Installation & Setup
bash
# 1. Clone the repository
git clone https://github.com/thedynasty23/Zomato-Restaurant-and-Reviews-Clustering.git
cd Zomato-Restaurant-and-Reviews-Clustering

# 2. Install dependencies
pip install pandas==2.2.2 numpy==1.26.4 scikit-learn==1.6.1
pip install matplotlib seaborn plotly wordcloud nltk textblob
pip install statsmodels jupyter

# 3. Launch Jupyter Notebook
jupyter notebook Zomato_Restaurant_Clustering.ipynb

# 4. Run all cells for complete analysis
Usage Examples
python
# Load and explore data
import pandas as pd
df_restaurant = pd.read_csv('Zomato Restaurant names and Metadata.csv')
df_reviews = pd.read_csv('Zomato Restaurant reviews.csv')

print(f"Restaurants: {df_restaurant.shape}")  # Output: 105
print(f"Reviews: {df_reviews.shape}")         # Output: 10000

# Price analysis
restaurant['Cost'] = restaurant['Cost'].str.replace(',', '').astype('int64')
top_expensive = restaurant.sort_values('Cost', ascending=False)[['Name', 'Cost']][:5]

# Clustering implementation
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(features_scaled)
💡 Business Insights & Recommendations
For Restaurant Owners
Quality over Price: High-rated restaurants (AB's - 4.88⭐) can charge premium prices (₹1,500)

Cuisine Strategy: Focus on North Indian and Chinese cuisines (highest demand)

Review Management: Engage with influential reviewers (13,000+ followers impact ratings)

Service Standards: Low-rated restaurants need immediate quality improvements

For Zomato Platform
Restaurant Onboarding: Target mid-range (₹500-1,200) segment - largest market

Marketing Strategy: Promote "Great Buffets" and "Food Hygiene" certified restaurants

Quality Assurance: Monitor restaurants with <3.0 ratings for intervention

Expansion: Limited competition in premium dining (₹2,000+) segment

For Customers
Value for Money: Paradise (₹800, 4.70⭐) offers excellent quality-price ratio

Premium Experience: AB's - Absolute Barbecues for special occasions

Budget Options: Amul (₹150) provides good quality at lowest prices

Cuisine Diversity: Gachibowli offers 25+ different cuisine types

🔧 Technical Implementation Details
Data Preprocessing Pipeline
python
# 1. Data Cleaning
- Remove duplicates: 36 rows
- Handle missing values: Collections (dropped), Timings (mode), Reviews (custom)
- Data type conversion: Cost (str→int), Rating (obj→float)

# 2. Feature Engineering  
- Extract reviewer metadata: followers, total_reviews
- Create time features: year, month, hour
- Cuisine count per restaurant: 1-6 cuisines

# 3. Text Processing (NLP)
- Tokenization and normalization
- Stopwords removal 
- Lemmatization
- TF-IDF vectorization
- Bag of Words encoding

# 4. Outlier Detection
- Isolation Forest algorithm
- Contamination rate: 0.01
- Features: Cost, Reviewer_Followers
Model Evaluation Metrics
python
# Clustering Quality Assessment
from sklearn.metrics import silhouette_score, davies_bouldin_score

# K-Means Results
silhouette_avg = 0.247  # Moderate cluster separation
davies_bouldin = 1.151  # Good cluster definition
calinski_harabasz = 20.115  # Excellent cluster separation

# Optimal K Selection
elbow_method_k = 5      # Within-cluster sum of squares analysis
silhouette_method_k = 5 # Average silhouette score analysis
📱 Sample Code Snippets
Restaurant Analysis
python
# Find restaurants sharing same price point
def analyze_price_distribution(df):
    price_groups = df.groupby('Cost')['Name'].apply(list).reset_index()
    price_groups['Count'] = price_groups['Name'].apply(len)
    return price_groups.sort_values('Count', ascending=False)

# Result: 13 restaurants at ₹500 price point
Review Sentiment Analysis
python
# Extract top reviewers by influence
def get_top_reviewers(df, n=5):
    return df.groupby('Reviewer').agg({
        'Reviewer_Total_Review': 'max',
        'Reviewer_Followers': 'max', 
        'Rating': 'mean'
    }).sort_values('Reviewer_Followers', ascending=False).head(n)
Clustering Implementation
python
# Complete clustering pipeline
def restaurant_clustering(features, n_clusters=5):
    # Standardization
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA dimensionality reduction
    pca = PCA(n_components=0.95)  # 95% variance retention
    features_pca = pca.fit_transform(features_scaled)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_pca)
    
    return labels, kmeans, pca, scaler
🎯 Future Enhancements
Short-term (Next 3 months)
 Real-time sentiment analysis dashboard

 Restaurant recommendation API

 Mobile app integration

 Advanced NLP with BERT/GPT models

Medium-term (6 months)
 Geographic clustering with location data

 Time-series analysis for seasonal trends

 Deep learning models for review classification

 Integration with live Zomato API

Long-term (1 year)
 Multi-city expansion analysis

 Predictive modeling for restaurant success

 Computer vision for food image analysis

 Real-time recommendation system

🤝 Contributing
We welcome contributions! Please follow these steps:

bash
# 1. Fork the repository
# 2. Create feature branch
git checkout -b feature/AmazingFeature

# 3. Commit changes
git commit -m 'Add some AmazingFeature'

# 4. Push to branch  
git push origin feature/AmazingFeature

# 5. Open Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author & Contact
thedynasty23

🔗 GitHub: @thedynasty23

📧 Project Link: Zomato-Restaurant-and-Reviews-Clustering

💼 LinkedIn: Connect with me

🙏 Acknowledgments
Data Source: Zomato restaurant data from Hyderabad, Gachibowli area

Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn communities

Inspiration: Food tech industry and data-driven restaurant insights

Academic Support: Machine Learning and NLP research communities

📊 Project Statistics
Development Time: 3 weeks

Lines of Code: 2,500+ lines

Data Points Analyzed: 10,000+ reviews, 105 restaurants

Features Engineered: 17 meaningful features

Visualizations Created: 15+ comprehensive charts

Machine Learning Models: 2 clustering algorithms

Statistical Tests: 3 hypothesis validations

🍕 Built with passion for food tech and data science! 📊
