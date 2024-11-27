from pathlib import Path
import numpy as np
from typing import Iterator, Tuple, List

class DataLoader:
    def __init__(self, image_folder: Path):
        self.image_folder = image_folder

    def walk_dir(path: Path) -> Iterator:
        """loops recursively through a folder

        Args:
            path (Path): folder to loop trough. If a directory
                is encountered, loop through that recursively.

        Yields:
            Generator: all paths in a folder and subdirs.
        """

        for p in Path(path).iterdir():
            if p.is_dir():
                yield from walk_dir(p)
                continue
            # resolve works like .absolute(), but it removes the "../.." parts
            # of the location, so it is cleaner
            yield p.resolve()

    # Gebruik de functie en print de resultaten
    for file_path in walk_dir(image_folder):
        print(file_path)