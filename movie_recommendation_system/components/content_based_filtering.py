from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendations(movies, title):
    tfidf = TfidfVectorizer(stop_words='english')
    movies['description'] = movies['genres'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies['description'])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]
