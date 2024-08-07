from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV

def collaborative_filtering(data):
    trainset, testset = train_test_split(data, test_size=0.25)
    
    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    return algo, predictions

def tune_parameters(data):
    param_grid = {
        'n_epochs': [20, 30],
        'lr_all': [0.002, 0.005],
        'reg_all': [0.4, 0.6]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)
    return gs.best_estimator['rmse']
