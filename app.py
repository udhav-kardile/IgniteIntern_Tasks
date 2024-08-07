from surprise import Dataset, Reader
from data_loader import load_data
from collaborative_filtering import collaborative_filtering, tune_parameters
from content_based_filtering import content_based_recommendations
from evaluation import evaluate

def main():
    movies, ratings = load_data()
    
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    
    algo, predictions = collaborative_filtering(data)
    rmse = evaluate(predictions)
    print(f'Collaborative Filtering RMSE: {rmse}')
    
    best_algo = tune_parameters(data)
    print(f'Best Collaborative Filtering Algorithm: {best_algo}')
    
    recommendations = content_based_recommendations(movies, 'Toy Story (1995)')
    print(f'Content-Based Recommendations for "Toy Story (1995)":\n{recommendations}')

if __name__ == "__main__":
    main()
