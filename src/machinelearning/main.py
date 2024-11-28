from pathlib import Path
import numpy as np
from typing import Iterator, Tuple, List
import mads_datasets
mads_datasets.__version__

from filehandling import Dataloader

dataloader = DataLoader(Path("data/images"))

for file_path in walk_dir(image_folder):
        print(file_path)

# src/mymodule/main.py
def main():
    print("Hello World")

if __name__ == '__main__':
    main()