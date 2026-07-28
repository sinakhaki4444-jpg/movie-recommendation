from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import streamlit as st
import os
@st.cache_data
def download_data():
    if not os.path.exists("ml-100k"):
        url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

        with urlopen(url) as zurl:
            with ZipFile(BytesIO(zurl.read())) as zfile:
                zfile.extractall('.')

    return True

download_data()
import pandas as pd


@st.cache_data
def load_data():

    ratings = pd.read_csv(
        'ml-100k/u.data',
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'unix_timestamp']
    )

    movies = pd.read_csv(
        'ml-100k/u.item',
        sep='|',
        usecols=range(2),
        names=['movie_id', 'title'],
        encoding='latin-1'
    )

    return ratings, movies


ratings, movies = load_data()

ratings = ratings[ratings.rating >= 4]

top_movies = (
    ratings.groupby("movie_id")
    .size()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

top_movies = top_movies.merge(
    movies,
    on="movie_id"
)


st.subheader("🎬 Top 10 Movies")

st.dataframe(
    top_movies[["title"]],
    use_container_width=True,
    hide_index=True
)





def recommend(movie):
    movie_id = str(movies[movies.title == movie].movie_id.values[0])

    results = []

    for id, score in model.wv.most_similar(movie_id)[:5]:
        title = movies[movies.movie_id == int(id)].title.values[0]

        results.append({
            "Movie": title,
            "Similarity": round(score, 2)
        })

    return pd.DataFrame(results)

movie_titles = movies['title'].tolist()

from rapidfuzz import process


def search_movies(text):
    results = process.extract(
        text,
        movie_titles,
        limit=5
    )

    return [r[0] for r in results]



from gensim.models import Word2Vec
import base64

@st.cache_data
def load_background(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()
@st.cache_resource
def load_model():
    return Word2Vec.load("movie_model.model")

model = load_model()

def set_background(image_file):
    data = load_background(image_file)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{data}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )



set_background("background.png.jpeg")

st.title("Movie Recommendation System My Mahsa")

st.markdown("""
You’re my soul mate Mahsa.
I can’t just handle that, and I think we made for each other.
 You are my everything. Just hold me and I believe in us.
 I know we belong together, I love you so much.
""")

query = st.text_input("Search movie:")

if query:

    results = search_movies(query)

    selected = st.selectbox(
        "Choose movie:",
        results
    )

    if st.button("Recommend"):

        similar_movies = recommend(selected)

        st.subheader("🔥 Similar Movies")

        st.dataframe(
            similar_movies,
            use_container_width=True,
            hide_index=True
        )

