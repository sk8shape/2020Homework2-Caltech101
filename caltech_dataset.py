from torchvision.datasets import VisionDataset
from sklearn.model_selection import train_test_split
from PIL import Image

import os
import os.path
import sys
import numpy as np

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):



    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        self.images_dataset = []
        self.labels = []
        self.labels_indx = []
        self.unique_labels = []

        file_path = "Caltech101" + "/" + self.split + ".txt"
        print (file_path)
        with open(file_path, "r" ) as fp:
            for line in fp:
                row = line.strip("\n")
                label_tmp = row.split("/")[0]
                if (label_tmp != "BACKGROUND_Google"):
                    img = pil_loader(root + "/" + row)
                    self.images_dataset.append(img)
                    self.labels.append(label_tmp)

        self.unique_labels = np.unique(self.labels)
        for lab in self.labels:
            self.labels_indx.append(list(self.unique_labels).index(lab))


    def __getitem__(self, index):
        image, label =   self.images_dataset[index], self.labels_indx[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        length = len(self.images_dataset)
        return length

    def split(self, ratio):

        tmp_train = []
        tmp_val = []
        train_set = []
        val_set = []

        for ul in range(len(self.unique_labels)):
            tmp = []
            for li in self.labels_indx:
                if(li == ul):
                    tmp.append(li)
            tmp_train, tmp_val = train_test_split(tmp, test_size = ratio)
            train_set.extend(tmp_train)
            val_set.extend(tmp_val)

        return train_set, val_set
