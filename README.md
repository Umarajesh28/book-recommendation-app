# Audible Insights: Intelligent Book Recommendation System

## Project Overview
Audible Insights is an intelligent book recommendation system that helps users discover books based on their preferences. The system leverages **NLP, clustering, and machine learning models** to provide content-based and clustering-based recommendations.

## Features
- **Content-Based Filtering**: Recommends books based on book descriptions and genres.
- **Clustering-Based Recommendations**: Groups books based on similarity using machine learning clustering.
- **Interactive Visualizations**: Provides insights into book trends, popular genres, and highly-rated authors.
- **Web Application**: Built using **Streamlit**, making it easy to use.
- **AWS Deployment**: Hosted on an **AWS EC2 instance** for accessibility and scalability.

## Project Structure
```
book_app/
│── models/
│   ├── content_similarity_matrix.pkl
│   ├── kmeans_clustering_model.pkl

│── Audible_Book_Data_preparation_Eda.ipynb
│──  Audible_nlp_clustering.ipynb
│──  Audible_Recommendation_System.ipynb
│── app.py  # Streamlit web application
│── books_clusters.csv  # Processed dataset
│── requirements.txt  # Required dependencies
│── README.md  # Project documentation
```

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Umarajesh28/book-recommendation-app.git
cd book-recommendation-app
```

### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows, use: myenv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Exploratory Data Analysis (EDA)
The **Audible_Book_Data_preparation_Eda.ipynb** notebook contains an in-depth analysis of:
- Most popular genres
- Highly-rated authors
- Relationship between reviews and book ratings
- Trends in book publication and reader preferences

## 📚 Recommendation Models
### 1️⃣ Content-Based Filtering
- Uses **TF-IDF vectorization and cosine similarity** to recommend books based on content descriptions.

### 2️⃣ Clustering-Based Recommendations
- Books are grouped into **clusters using K-Means**, and recommendations are provided based on cluster similarity.

##  Deployment on AWS EC2
The application is deployed on an **AWS EC2 instance** running **Ubuntu**.
### **Steps to Deploy:**
1. **Launch an EC2 instance** with Ubuntu 20.04.
2. **SSH into the instance** and install required dependencies.
3. **Transfer project files** using SCP or Git.
4. **Run Streamlit app** and allow inbound traffic on port 8501.

##  Results & Business Use Cases
- **Personalized Reading Experience**: Users get tailored book recommendations based on their preferences.
- **Enhanced Library & Bookstore Systems**: Libraries and bookstores can improve book recommendations for users.
- **Reader Engagement**: Suggests trending and highly-rated books to boost engagement.



