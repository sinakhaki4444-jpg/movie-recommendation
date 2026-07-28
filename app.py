from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

with urlopen(url) as zurl:
    with ZipFile(BytesIO(zurl.read())) as zfile:
        zfile.extractall('.')

import pandas as pd

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


ratings = ratings[ratings.rating >= 4]



from collections import defaultdict

pairs = defaultdict(int)

for group in ratings.groupby("user_id"):
    user_movies = list(group[1]["movie_id"])

    for i in range(len(user_movies)):
        for j in range(i + 1, len(user_movies)):
            pairs[(user_movies[i], user_movies[j])] += 1

import networkx as nx

G = nx.Graph()

for pair in pairs:
    movie1, movie2 = pair
    score = pairs[pair]

    if score >= 20:
        G.add_edge(movie1, movie2, weight=score)

from gensim.models import Word2Vec

model = Word2Vec.load("movie_model.model")


def recommend(movie):
    movie_id = str(movies[movies.title == movie].movie_id.values[0])

    for id in model.wv.most_similar(movie_id)[:5]:
        title = movies[movies.movie_id == int(id[0])].title.values[0]
        print(f'{title}: {id[1]:.2f}')

movie_titles = movies['title'].tolist()

from rapidfuzz import process


def search_movies(text):
    results = process.extract(
        text,
        movie_titles,
        limit=5
    )

    return [r[0] for r in results]


import streamlit as st
import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()

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
import os


set_background("background.png.jpeg")

st.title("Movie Recommendation System")


query = st.text_input(
    "Search movie:"
)


if query:

    results = search_movies(query)

    selected = st.selectbox(
        "Choose movie:",
        results
    )


    if st.button("Recommend"):

        recommend(selected)


