import os
import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset


class Cub200DataSet(Dataset):
    data_folder = 'CUB_200_2011'
    image_folder = os.path.join(data_folder, 'images')
    attribute_folder = os.path.join(data_folder, 'attributes')
    parts_folder = os.path.join(data_folder, 'parts')

    def __init__(self, root, create=True):
        self.root = os.path.expanduser(root)

        if create:
            self._create_metadata()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, self.data_folder, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        labels = pd.read_csv(os.path.join(self.root, self.data_folder, 'image_class_labels.txt'),
                             sep=' ', names=['img_id', 'class_id'])
        attributes = pd.read_csv(os.path.join(self.root, self.attribute_folder, 'image_attribute_labels.txt'),
                                 sep=' ', names=['img_id', 'attribute_id', 'is_present', 'certainty', 'time_spent'])
        label_names = pd.read_csv(os.path.join(self.root, self.data_folder, 'classes.txt'),
                                  sep=' ', names=['class_id', 'class_name'])
        attribute_names = pd.read_csv(os.path.join(self.root, 'attributes.txt'),
                                      sep=' ', names=['attribute_id', 'attribute_name'])

        present_attributes = attributes.loc[(attributes['is_present'] == 1) & (attributes['certainty'] >= 3)]
        present_attributes = present_attributes.groupby('img_id')["attribute_id"].apply(np.array).reset_index()

        label_map = dict(zip(label_names.class_id, label_names.class_name))
        attribute_map = dict(zip(attribute_names.attribute_id, attribute_names.attribute_name))
        self.data = images.merge(labels, on='img_id').merge(present_attributes, on='img_id')
        self.label_map = label_map
        self.attribute_map = attribute_map

    def pre_process_labels_and_annotations(self, label):
        numbers = r'[0-9]'
        punctuation = r'([^\w\s]|_)'
        numbers_removed = re.sub(numbers, '', label)
        punctuation_removed = re.sub(punctuation, ' ', numbers_removed)
        return punctuation_removed

    def get_label_map(self):
        return self.label_map

    def get_attribute_map(self):
        return self.attribute_map

    def _create_metadata(self):
        self._load_metadata()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        true_path = os.path.join(self.root, self.image_folder, sample.filepath)

        return {"path": true_path, "id": sample.img_id, "label": sample.class_id, "attributes": sample.attribute_id}
