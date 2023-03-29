from glob import glob

from PIL import Image
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, image_queries, transform=None):
        self.image_queries = image_queries
        self.transform = transform
        self.filepaths = self.get_filepaths()

    def get_filepaths(self):
        filepaths = [filepath for query in self.image_queries for filepath in glob(query)]
        return filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Get the queried filepath
        filepath = self.filepaths[idx]

        # Read the image
        image = Image.open(filepath).convert('RGB')
        return self.transform(image), 0
