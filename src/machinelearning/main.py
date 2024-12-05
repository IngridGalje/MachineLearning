from pathlib import Path
import numpy as np
from typing import Iterator, Tuple, List
import mads_datasets
mads_datasets.__version__

from machinelearning.filehandling import DataLoader
from machinelearning.visualisations.heatmap import HeatmapRuns
from mads_datasets.settings import FileTypes
from mads_datasets import DatasetFactoryProvider, DatasetType
import logging
import pandas as pd
import os

############ 0. Logging niveau en folder
logging.basicConfig(filename='machinelearning.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

############ 1. Read data 
curr_folder = Path(os.path.abspath(__file__)).parent.parent.parent
path = curr_folder / ".dev/runs.csv"
df = pd.read_csv(path, sep=',')
pd.set_option("display.max_columns", None)
logging.info("Gegevens zijn ingelezen.")

def heatmap(df):
    heatmap = HeatmapRuns(df)
    heatmap.plot_corr_matrix()

def main():
    heatmap(df)

if __name__ == '__main__':
    main()