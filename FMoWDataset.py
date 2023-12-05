import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = 327279692

class FMoWDataset(Dataset):
    def __init__(self, df, preprocess):
        self.df = df
        self.preprocess = preprocess

        self.image_paths = df['file_path'].tolist()
        self.labels = df['category'].tolist()
        self.textual_inputs = self.create_textual_inputs(df)

        self.label_to_idx = {label: idx for idx, label in enumerate(np.unique(self.labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def create_textual_inputs(self, df):
        textual_inputs = []
        for _, row in df.iterrows():
            text = f"a satellite image of a {row['category']} in {row['country']} in the {row['ns-hemisphere'][:-3]} {row['ew-hemisphere']} hemisphere of {row['continent']}"
            textual_inputs.append(text)
        return textual_inputs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self.preprocess(Image.open(img_path))
        label_idx = self.label_to_idx[self.labels[idx]]
        text = self.textual_inputs[idx]
        return image, text, label_idx
