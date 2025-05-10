# 🛍️ End-to-End Recommendation System for E-commerce

This project is an end-to-end hybrid recommendation system for an e-commerce platform. It combines collaborative filtering, content-based filtering, and hybrid techniques to generate personalized product recommendations.

## 📁 Project Structure

├── Dataset/
│ ├── events.csv
│ ├── item_properties_part1.csv
│ └── item_properties_part2.csv
├── project.py # Main backend logic for training and recommendation
├── App.py # Command-line interface for user interaction
├── item_metadata.csv # Preprocessed item metadata
├── events_clean.csv # Preprocessed events data
└── README.md


## 🧠 Features

- 📊 **Data Preprocessing:** Cleans and merges event and item property data.
- 🧮 **Collaborative Filtering:** Uses the SVD algorithm from the `surprise` library.
- 🧾 **Content-Based Filtering:** Uses TF-IDF and cosine similarity on item metadata.
- 🤝 **Hybrid Recommender:** Blends collaborative and content scores.
- 🔍 **CLI Interface:** Lets users input a user ID and get recommendations via the terminal.

## ⚙️ How It Works

### Step 1: Preprocessing
- Merges item metadata from two datasets.
- Filters for meaningful events (e.g., `view`).
- Extracts latest item properties and pivots them.

### Step 2: Collaborative Filtering
- Trains an SVD model on user-item interaction data.
- Evaluates model performance using RMSE.

### Step 3: Content-Based Filtering
- Builds a TF-IDF matrix from item category metadata.
- Calculates cosine similarity between items.

### Step 4: Hybrid Recommendation
- Combines collaborative and content scores using a weighted average.
- Generates top-N recommendations for a given user.

## ▶️ Running the Project

### Step 1: Install Dependencies
Make sure you have the required libraries installed:

.

📌 Notes
Dataset files are large, so Git LFS is used to manage them.

events_clean.csv and item_metadata.csv are saved after preprocessing to avoid redundant computation.

🙌 Acknowledgements
[Retailrocket Recommender Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

surprise library by Nicolas Hug
