import os
from pathlib import Path

project_name = "movie_recommendation_system"

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_loader.py",
    f"{project_name}/components/collaborative_filtering.py",
    f"{project_name}/components/content_based_filtering.py",
    f"{project_name}/components/evaluation.py",
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    f"{project_name}/data/movies.csv",
    f"{project_name}/data/ratings.csv",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "README.md",
    "setup.py",
    "config/model.yaml",
    "config/schema.yaml",
    "notebooks/EDA.ipynb",
    "notebooks/collaborative_filtering.ipynb",
    "notebooks/content_based_filtering.ipynb",
    "notebooks/evaluation.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")
