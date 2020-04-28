from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

class data_elem:
    def __init__(self, image, label):
        self.sample = image
        self.lbl = label

    def get_sample(self):
        return self.sample
    def get_label(self):
        return self.lbl


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):

    my_dataset = []
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        split_path = root + "/" + self.split + ".txt"
        file_path = root.split("/")[0] + "/" + self.split + ".txt"
        with open(split_path, "r" ) as sp:
            row = sp.read()
            img = pil_loader(split_path+"/"+row)
            item = data_elem(img,row.split("/")[0])
            my_dataset.append(item)

    def __getitem__(self, index):

        image, label =   my_dataset[index].get_image(), my_dataset[index].get_label()

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        length = len(my_dataset)
        return length
