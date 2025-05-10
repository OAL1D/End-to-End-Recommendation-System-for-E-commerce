import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Title
st.set_page_config(page_title="E-commerce Recommender", layout="wide")
st.title("🛒 Recommendation System")

# Load Data
@st.cache_data
def load_data():
    events = pd.read_csv("events_clean.csv")
    item_metadata = pd.read_csv("item_metadata.csv")

    # Combine 'category_id' for TF-IDF if available
    if 'category_id' in item_metadata.columns:
        item_metadata['combined'] = item_metadata['category_id'].astype(str)
    return events, item_metadata

events, item_metadata = load_data()

# Train collaborative filtering model (or load if saved)
@st.cache_resource
def train_model(events):
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(events[['user_id', 'item_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    return model

model = train_model(events)

# Prepare TF-IDF similarity matrix (limit to top 1000 items)
@st.cache_resource
def prepare_similarity_matrix(item_metadata, top_n=1000):
    limited_items = item_metadata.head(top_n)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(limited_items['combined'])
    sim_matrix = cosine_similarity(tfidf_matrix)
    item_ids = limited_items['item_id'].values
    item_id_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    return sim_matrix, item_id_to_index, item_ids

sim_matrix, item_id_to_index, limited_item_ids = prepare_similarity_matrix(item_metadata)

# Input from user
user_id = st.number_input("🔢 Enter your User ID", min_value=1, value=1, step=1)

if st.button("🔍 Get Recommendations"):
    user_items = events[events['user_id'] == user_id]['item_id'].unique()
    all_items = events['item_id'].unique()
    items_to_predict = [item for item in all_items if item not in user_items]

    if not items_to_predict:
        st.warning("⚠️ No items to recommend. Try another user ID.")
    else:
        top_recs = []
        for item in items_to_predict:
            # Collaborative score
            collab_score = model.predict(user_id, item).est

            # Content-based score
            if item in item_id_to_index:
                similarities = []
                for seen_item in user_items:
                    if seen_item in item_id_to_index:
                        sim = sim_matrix[item_id_to_index[item]][item_id_to_index[seen_item]]
                        similarities.append(sim)
                content_score = np.mean(similarities) if similarities else 0
            else:
                content_score = 0

            # Hybrid score
            final_score = 0.7 * collab_score + 0.3 * content_score
            top_recs.append((item, final_score))

        top_recs = sorted(top_recs, key=lambda x: x[1], reverse=True)[:10]

        st.subheader(f"📌 Top 10 Recommendations for User {user_id}")
        for idx, (item, score) in enumerate(top_recs, start=1):
            category = item_metadata[item_metadata['item_id'] == item]['category_id'].values[0] if 'category_id' in item_metadata.columns else 'Unknown'
            st.markdown(f"**{idx}. Item ID:** `{item}` | **Score:** {score:.2f} | **Category:** `{category}`")

