import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os

class HeatmapRuns:
    def __init__(self, df) -> None:
        self.df = df
        self.images_dir = Path(os.path.abspath(__file__)).parent.parent.parent.parent/'img'
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def clear_close(self):
        """Clears the current figure and axes."""
        plt.clf()
        plt.close()

    def plot_corr_matrix(self):
        self.clear_close()  # Clear the figure and axes before plotting

        columns_numeric = self.df.select_dtypes(include=[float, int])
        correlation_matrix = columns_numeric.corr()

        # Create a heatmap with annotations
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title("Correlation Matrix")
        plt.show()
        plt.savefig(self.images_dir/'heatmap.png')