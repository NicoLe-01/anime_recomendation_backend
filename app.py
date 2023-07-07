from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from flask_cors import CORS


cosine_sim_df = pd.read_csv('tfidf_matrix.csv')
animes = pd.read_csv('animes_preprocessed.csv')


def anime_recommendation(nama_anime, similarity_data=cosine_sim_df, items=animes[['name', 'genre']], k=5):
  index = similarity_data.loc[:, nama_anime].to_numpy().argpartition(
      range(-1, -k, -1)
  )

  closest = similarity_data.columns[index[-1:-(k+2):-1]]
  closest = closest.drop(nama_anime, errors='ignore')
  
  return closest.to_list()

app = Flask(__name__)
CORS(app)

# Endpoint for recommendation
@app.route('/recommend/<title>', methods=['GET'])
def recommend_anime(title):
    recommendations = anime_recommendation(title, similarity_data=cosine_sim_df, items=animes[['name', 'genre']], k=5)
    return jsonify({'recommendations': recommendations})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
#This server slowing down little bit