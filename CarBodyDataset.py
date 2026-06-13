from torch.utils.data import Dataset
import json
import os
from PIL import Image

class CarBodyDataset(Dataset):
    def __init__(self, data_dir, mapping_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # reading the mapping file
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.folder_to_category = json.load(f)

        # list of unique car types
        self.categories = sorted(list(set(self.folder_to_category.values())))
        # class (text) -> values (num)
        self.class_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        self.image_paths = []
        self.labels = []

        # appends category number for each picture paths
        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if os.path.isdir(folder_path):
                category = self.folder_to_category.get(folder_name)
                label_idx = self.class_to_idx[category]

                for img_name in os.listdir(folder_path):
                    self.image_paths.append(os.path.join(folder_path, img_name))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
