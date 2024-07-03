from flask import Flask, render_template, request
import pandas as pd
from rapidfuzz import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser

app = Flask(__name__)

# Load the CSV file into a DataFrame
df = pd.read_csv('Music_Info.csv')

# Ensure 'tags' column exists and is not empty
if 'tags' not in df.columns or df['tags'].isnull().all():
    raise ValueError("The 'tags' column is missing or empty in the DataFrame.")

# Define a function to preprocess tags lists for TF-IDF
def preprocess_tags(tags_str):
    if pd.isna(tags_str):
        return ""
    return str(tags_str).replace(", ", " ")  # Replace commas with spaces

# Preprocess tags data
df['tags_text'] = df['tags'].apply(preprocess_tags)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit TF-IDF vectorizer to the tags text data
tags_vectors = vectorizer.fit_transform(df['tags_text'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    song_name = request.form['song_name'].lower()

    # First search for songs where the title starts with the input
    song_row_starts_with = df[df['name'].str.lower().str.startswith(song_name, na=False)]

    # If no matches are found, search for songs where the title contains the input
    if not song_row_starts_with.empty:
        song_list = song_row_starts_with.head(3)
    else:
        song_row_contains = df[df['name'].str.lower().str.contains(song_name, na=False)]
        if not song_row_contains.empty:
            song_list = song_row_contains.head(3)
        else:
            titles = df['name'].tolist()
            similar_titles = process.extract(song_name, titles, limit=10)
            song_list = pd.concat([df[df['name'] == title] for title, score in similar_titles])

    return render_template('results.html', song_list=song_list, song_name=song_name)

@app.route('/play/<int:song_index>')
def play(song_index):
    selected_song_row = df.iloc[song_index]
    spotify_url = selected_song_row['spotify_preview_url']

    # Open the Spotify URL in the web browser
    webbrowser.open(spotify_url)

    # Get the played song's tags text
    played_song_tags_text = df.loc[song_index, 'tags_text']

    # Convert tags text to TF-IDF vector
    played_song_tags_vector = vectorizer.transform([played_song_tags_text])

    # Calculate cosine similarity between played song tags and other songs
    tags_similarities = cosine_similarity(played_song_tags_vector, tags_vectors)

    # Get indices of top 5 most similar songs (excluding the played song)
    top_similar_indices = tags_similarities.argsort()[0, -6:-1]

    # Gather similar songs
    similar_songs = [df.iloc[i] for i in reversed(top_similar_indices)]

    return render_template('similar.html', selected_song=selected_song_row, similar_songs=similar_songs)

if __name__ == '__main__':
    app.run(debug=True)
