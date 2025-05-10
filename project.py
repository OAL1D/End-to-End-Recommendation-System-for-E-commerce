import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load and preprocess data
print("\U0001F680 Loading and preprocessing data...")

# Load raw data
events = pd.read_csv(r"E:\End to end recommendation system for e-commerce\Dataset\events.csv")
props1 = pd.read_csv(r"E:\End to end recommendation system for e-commerce\Dataset\item_properties_part1.csv")
props2 = pd.read_csv(r"E:\End to end recommendation system for e-commerce\Dataset\item_properties_part2.csv")

# Combine properties
props = pd.concat([props1, props2])
del props1, props2

# Extract latest property values
props = props.sort_values('timestamp').drop_duplicates(subset=['itemid', 'property'], keep='last')
props_pivot = props.pivot(index='itemid', columns='property', values='value').reset_index()

# Debugging: Print column names to check if 'categoryid' exists
print(f"Columns in props_pivot: {props_pivot.columns}")

# Extract metadata safely
metadata_cols = ['itemid']
if 'categoryid' in props_pivot.columns:
    metadata_cols.append('categoryid')

item_metadata = props_pivot[metadata_cols].rename(columns={'itemid': 'item_id'})
if 'categoryid' in item_metadata.columns:
    item_metadata.rename(columns={'categoryid': 'category_id'}, inplace=True)

# Fill missing category_id with 'Unknown' if the column exists
if 'category_id' in item_metadata.columns:
    item_metadata['category_id'] = item_metadata['category_id'].fillna('Unknown')
else:
    print("Warning: 'category_id' column is missing in item_metadata. Defaulting to 'Unknown'.")

# Combine 'category_id' for item metadata (only if it exists)
if 'category_id' in item_metadata.columns:
    item_metadata['combined'] = item_metadata['category_id'].astype(str)
else:
    print("Warning: 'category_id' column is missing. Content-based filtering will not work properly.")

# Filter events for relevant interactions
events = events[events['event'] == 'view']
events = events[['visitorid', 'itemid', 'timestamp']].rename(columns={'visitorid': 'user_id', 'itemid': 'item_id'})
events['rating'] = 1

# Save preprocessed data
item_metadata.to_csv("item_metadata.csv", index=False)
events.to_csv("events_clean.csv", index=False)

print("\u2705 Preprocessing complete!")

# Step 2: Train Collaborative Filtering Model
print("\U0001F4BB Training collaborative filtering model...")
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(events[['user_id', 'item_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)
predictions = model.test(testset)

# Evaluate
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse:.4f}")
print("\u2705 Model evaluation complete. RMSE:")
print(f"RMSE: {rmse:.4f}")

# Step 3: Content-Based Filtering using TF-IDF
print("\U0001F9EE Preparing content-based filtering...")

# Ensure that 'combined' column is created only if 'category_id' exists
if 'combined' in item_metadata.columns:
    # Limit to top 1000 items (for example) to reduce memory usage
    limited_items = item_metadata.head(1000)
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(limited_items['combined'])
    sim_matrix = cosine_similarity(tfidf_matrix)
    item_ids = limited_items['item_id'].values
    item_id_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

else:
    print("⚠️ Unable to generate content-based filtering due to missing 'category_id' column.")

# Step 4: Hybrid Recommendations
print("\U0001F3AF Generating recommendations...")
user_input = input("\U0001F50D Enter a User ID to get recommendations (or press Enter to skip): ")

if user_input:
    user_id = int(user_input)
    user_items = events[events['user_id'] == user_id]['item_id'].unique()
    all_items = events['item_id'].unique()
    items_to_predict = [item for item in all_items if item not in user_items]

    if not items_to_predict:
        print(f"⚠️  No recommendations found for user ID {user_id}.")
    else:
        top_n = 10
        top_recs = []

        for item in items_to_predict:
            # Collaborative prediction
            collab_score = model.predict(user_id, item).est

            # Content-based prediction
            if item in item_id_to_index:
                similarities = []
                for seen_item in user_items:
                    if seen_item in item_id_to_index:
                        sim = sim_matrix[item_id_to_index[item]][item_id_to_index[seen_item]]
                        similarities.append(sim)
                content_score = np.mean(similarities) if similarities else 0
            else:
                content_score = 0

            # Hybrid score (weighted average)
            final_score = 0.7 * collab_score + 0.3 * content_score
            top_recs.append((item, final_score))

        top_recs = sorted(top_recs, key=lambda x: x[1], reverse=True)[:top_n]

        print(f"\n\U0001F4E3 Top {top_n} Hybrid Recommendations for User {user_id}:")
        for idx, (item, rating) in enumerate(top_recs, 1):
            category = item_metadata[item_metadata['item_id'] == item]['category_id'].values[0] if 'category_id' in item_metadata.columns else 'Unknown'
            print(f"{idx}. Item ID: {item} | Score: {rating:.2f} | Category: {category}")
else:
    print("⏳ Skipped generating user recommendations.")
