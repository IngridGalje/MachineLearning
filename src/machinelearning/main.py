from pathlib import Path
import numpy as np
from typing import Iterator, Tuple, List
import mads_datasets
mads_datasets.__version__

# Gebruik absolute import
from machinelearning.filehandling import DataLoader
from mads_datasets.settings import FileTypes

from mads_datasets import DatasetFactoryProvider, DatasetType

flowersfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
flowersfactory.download_data()

image_folder = flowersfactory.subfolder
print(image_folder)

def dataloader(image_folder: Path):
    dataloader = DataLoader(image_folder)
    dataloader.walk_dir(image_folder)
    dataloader.print_file_paths(limit=5)


#print filetypes
for ft in FileTypes:
    print(ft)    

# src/mymodule/main.py
def main():
    dataloader(image_folder)

if __name__ == '__main__':
    main()