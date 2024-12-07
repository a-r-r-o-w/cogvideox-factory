import pathlib

import pandas as pd
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    def __init__(self, input_file: str) -> None:
        self.input_file = pathlib.Path(input_file)

        assert self.input_file.is_file()

        df = pd.read_csv(input_file)
        self.filenames = df["filename"]
        self.summaries = df["summary"]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        return {
            "filename": self.filenames[index],
            "summary": self.summaries[index],
        }
