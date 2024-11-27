import os
from pathlib import Path

def make_img_dir():
    # Haal de huidige werkdirectory op
    curr_folder = os.getcwd()

    # Definieer het pad naar de img folder
    project_dir = Path(curr_folder) 
    images_dir = project_dir / 'img'
    images_dir.mkdir(parents=True, exist_ok=True)  # Maak de map aan als deze niet bestaat

def make_csv_dir():
    # Haal de huidige werkdirectory op
    curr_folder = os.getcwd()

    # Definieer het pad naar de img folder
    project_dir = Path(curr_folder) 
    analyses_dir = project_dir / 'data' / 'analyses'
    analyses_dir.mkdir(parents=True, exist_ok=True)  # Maak de map aan als deze niet bestaat
    # csv_file = analyses_dir / filename
    # df.to_csv(csv_file, index=False)