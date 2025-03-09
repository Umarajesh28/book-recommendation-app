import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import requests
from io import BytesIO

# Custom CSS for a dark, modern theme
st.markdown("""
    <style>
        .main {
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: 'Roboto', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        h1, h2, h3 {
            color: #ff6f61;
        }
        .stButton>button {
            background-color: #ff6f61;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #ff3b2f;
        }
        .stMarkdown {
            line-height: 1.8;
        }
        .stDataFrame {
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            background-color: #2d2d2d;
        }
        .stSelectbox, .stSlider {
            background-color: #2d2d2d;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Cache loading models and data to avoid reloading every time
@st.cache_data
def load_data():
    return pd.read_csv("/home/ubuntu/book_app/books_clusters.csv")

@st.cache_resource
def load_content_sim_matrix_model():
    with open("/home/ubuntu/book_app/models/content_similarity_matrix.pkl", "rb") as file:
        return pickle.load(file)

@st.cache_resource
def load_kmeans_model():
    with open("/home/ubuntu/book_app/models/kmeans_clustering_model.pkl", "rb") as file:
        return pickle.load(file)

# Load dataset and models
df = load_data()
content_sim_matrix_model = load_content_sim_matrix_model()
kmeans_model = load_kmeans_model()

# Function to fetch book covers from the web (using Google Books API)
def fetch_book_cover(book_title, author):
    try:
        response = requests.get(f"https://www.googleapis.com/books/v1/volumes?q=intitle:{book_title}+inauthor:{author}&maxResults=1")
        data = response.json()
        if data['totalItems'] > 0:
            return data['items'][0]['volumeInfo']['imageLinks']['thumbnail']
        else:
            return None
    except:
        return None

# Content-based recommendations for books within the selected genre
def recommend_books_by_content_in_genre(df, content_sim_matrix, genre, num_recs=5):
    genre_df = df[df['Genre'] == genre]
    if genre_df.empty:
        return []

    genre_idx = genre_df.index.to_list()
    content_sim_matrix_genre = content_sim_matrix[genre_idx][:, genre_idx]

    similar_books = list(enumerate(content_sim_matrix_genre[0]))
    sorted_books = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:num_recs+1]
    recommendations = [
        (genre_df.iloc[i[0]]['Book Name'], genre_df.iloc[i[0]]['Author'], genre_df.iloc[i[0]]['Rating'], genre_df.iloc[i[0]]['Number of Reviews'], genre_df.iloc[i[0]]['Processed_Description'], genre_df.iloc[i[0]]['Price'])
        for i in sorted_books
    ]
    return recommendations

# Clustering-based recommendations for books within the selected genre
def recommend_books_by_cluster_in_genre(df, genre, kmeans_model, num_recs=5):
    genre_df = df[df['Genre'] == genre]
    if genre_df.empty:
        return []

    book_cluster = genre_df['cluster'].mode()[0]
    cluster_books = genre_df[genre_df['cluster'] == book_cluster][['Book Name', 'Author', 'Rating', 'Number of Reviews', 'Processed_Description', 'Price']].head(num_recs).values.tolist()
    return cluster_books

# Truncate long descriptions
def truncate_description(description, max_length=200):
    return description[:max_length] + '...' if len(description) > max_length else description

def intro_page():
    st.title("📚 Welcome to the Book Discovery Hub!")
    
    # Add an image banner to make the page more engaging
    st.image("https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80", use_container_width=True)
    
    # Section to describe the purpose of the app
    st.markdown("""
        Welcome to the **Book Discovery Hub**, your ultimate destination for finding the perfect book! Whether you're a fan of thrillers, romance, or self-help, our system helps you discover books tailored to your tastes.

        **What can you do here?**
        - 📊 **Explore the Dataset**: Dive into the data and uncover trends like the most popular genres, highly-rated authors, and more.
        - 🔍 **Get Personalized Recommendations**: Use advanced **Content-Based** and **Clustering-Based** models to receive tailored book suggestions.
        - 📈 **Visualize Trends**: Interactive charts and graphs to help you understand the data better.

        🚀 **How it works:**
        - The system analyzes book features such as ratings, reviews, genres, and descriptions to provide personalized recommendations.
        - **Content-Based** recommendations use book similarity (based on descriptions) to suggest new books within your preferred genres.
        - **Clustering-Based** recommendations group books by similarities in content, providing insights into hidden gems or popular books in your chosen genre.

        🔧 **Tools at Your Disposal:**
        - **Interactive Plots**: Visualize the book trends with interactive charts and graphs.
        - **Customizable Recommendations**: Choose the number of recommendations and the genre that interests you.

        **Use the sidebar to navigate** between different sections and get started on your book discovery journey! ✨
    """, unsafe_allow_html=True)

def eda_faq_page():
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.markdown("""
        Explore the dataset and uncover insights about books, genres, authors, and more. Below, we answer key questions using interactive visualizations.
    """)

    # Question 1: Most popular genres
    st.subheader("1. What are the most popular genres in the dataset?")
    genre_count = df['Genre'].value_counts().head(10)
    fig = px.bar(genre_count, x=genre_count.index, y=genre_count.values, labels={'y': 'Book Count', 'index': 'Genres'},
                 title='Top 10 Most Popular Genres', color=genre_count.index, color_continuous_scale='Viridis')
    st.plotly_chart(fig)

    # Question 2: Highest-rated authors
    st.subheader("2. Which authors have the highest-rated books?")
    top_authors = df.groupby('Author').agg({'Rating': 'mean', 'Book Name': 'count'}).reset_index()
    top_authors = top_authors[top_authors['Book Name'] > 1].sort_values('Rating', ascending=False).head(10)
    fig = px.bar(top_authors, x='Author', y='Rating', color='Rating', title="Top 10 Authors by Average Rating",
                 labels={'Rating': 'Average Rating'}, color_continuous_scale='Plasma')
    st.plotly_chart(fig)

    # Question 3: Average rating distribution
    st.subheader("3. What is the average rating distribution across books?")
    fig, ax = plt.subplots()
    sns.histplot(df['Rating'], kde=True, ax=ax, color='#ff6f61')
    ax.set_title('Distribution of Ratings', color='white')
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # Question 4: Ratings vs. Review Counts
    st.subheader("4. How do ratings vary between books with different review counts?")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Number of Reviews', y='Rating', ax=ax, color='#ff6f61')
    ax.set_title('Ratings vs. Review Counts', color='white')
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # Question 5: Frequently clustered books
    st.subheader("5. Which books are frequently clustered together based on descriptions?")
    cluster_count = df['cluster'].value_counts().head(5)
    st.write("Top 5 Clusters by Book Count:")
    st.dataframe(cluster_count)

    # Question 6: Genre similarity and recommendations
    st.subheader("6. How does genre similarity affect book recommendations?")
    st.markdown("""
        Genre similarity plays a crucial role in content-based recommendations by grouping books with similar themes, topics, and styles. Books from the same genre tend to have higher cosine similarity scores, leading to stronger recommendations within the genre.
    """)

    # Question 7: Effect of author popularity on book ratings
    st.subheader("7. What is the effect of author popularity on book ratings?")
    author_popularity = df.groupby('Author').agg({'Rating': 'mean', 'Number of Reviews': 'sum'}).reset_index()
    fig = px.scatter(author_popularity, x='Number of Reviews', y='Rating', hover_data=['Author'],
                     title='Author Popularity vs. Ratings')
    st.plotly_chart(fig)

    # Question 8: Feature combinations for accurate recommendations
    st.subheader("8. Which combination of features provides the most accurate recommendations?")
    st.markdown("""
        Feature combinations such as **Genre**, **Ratings**, **Review Counts**, and **Author Popularity** can help fine-tune recommendations. By combining these features, we can better understand user preferences and optimize recommendations.
    """)

    # Question 9: Hidden gems (high-rated, low-popularity books)
    st.subheader("9. Identify books that are highly rated but have low popularity to recommend hidden gems.")
    hidden_gems = df[(df['Rating'] >= 4.5) & (df['Number of Reviews'] < 100)]
    st.dataframe(hidden_gems[['Book Name', 'Rating', 'Number of Reviews']])

def rec_system_page():
    st.title("🔍 Personalized Book Recommendations")

    st.sidebar.header("Customize Your Recommendations")
    rec_method = st.sidebar.radio("Choose Recommendation Method", ["Content-Based", "Clustering-Based"])
    genre = st.sidebar.selectbox("Choose a Genre", df['Genre'].unique())
    num_recs = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

    if rec_method == "Content-Based":
        st.header(f"📚 Content-Based Recommendations for {genre}")
        recs = recommend_books_by_content_in_genre(df, content_sim_matrix_model, genre, num_recs)
        if recs:
            for rec in recs:
                st.markdown(f"**Book Name:** {rec[0]}\n\n"
                            f"**Author:** {rec[1]}\n\n"
                            f"**Rating:** ⭐ {rec[2]}\n\n"
                            f"**Number of Reviews:** 📝 {rec[3]}\n\n"
                            f"**Price:** 💰 {rec[5]}\n\n"
                            f"**Description:** {truncate_description(rec[4])}\n\n")
                # Fetch and display book cover
                cover_url = fetch_book_cover(rec[0], rec[1])
                if cover_url:
                    st.image(cover_url, width=150)
                st.write("---")
        else:
            st.warning(f"No recommendations found for {genre}.")
    
    elif rec_method == "Clustering-Based":
        st.header(f"📊 Clustering-Based Recommendations for {genre}")
        recs = recommend_books_by_cluster_in_genre(df, genre, kmeans_model, num_recs)
        if recs:
            for rec in recs:
                st.markdown(f"**Book Name:** {rec[0]}\n\n"
                            f"**Author:** {rec[1]}\n\n"
                            f"**Rating:** ⭐ {rec[2]}\n\n"
                            f"**Number of Reviews:** 📝 {rec[3]}\n\n"
                            f"**Price:** 💰 {rec[5]}\n\n"
                            f"**Description:** {truncate_description(rec[4])}\n\n")
                # Fetch and display book cover
                cover_url = fetch_book_cover(rec[0], rec[1])
                if cover_url:
                    st.image(cover_url, width=150)
                st.write("---")
        else:
            st.warning(f"No recommendations found for {genre}.")

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Home", "Explore Data", "Get Recommendations"])

# Display selected page
if menu == "Home":
    intro_page()
elif menu == "Explore Data":
    eda_faq_page()
elif menu == "Get Recommendations":
    rec_system_page()
