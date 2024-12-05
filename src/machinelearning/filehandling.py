from pathlib import Path
from typing import Iterator

class DataLoader:
    def __init__(self, folder: Path):
        self.folder = folder

    def walk_dir(self, path: Path) -> Iterator[Path]:
        """loops recursively through a folder

        Args:
            path (Path): folder to loop through. If a directory
                is encountered, loop through that recursively.

        Yields:
            Generator: all paths in a folder and subdirs.
        """
        for p in Path(path).iterdir():
            if p.is_dir():
                yield from self.walk_dir(p)
                continue
            yield p.resolve()

    def print_file_paths(self, limit: int = 5):
        count = 0
        for file_path in self.walk_dir(self.folder):
            if count >= limit:
                break
            print(file_path)
            count += 1
