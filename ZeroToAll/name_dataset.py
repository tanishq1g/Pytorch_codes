import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
import gzip

class NameDataset(Dataset):
    
    # Initialize your data, download, etc.
    def __init__(self, is_train_set=False):
        filename = './data/names_train.csv.gz' if is_train_set else './data/names_test.csv.gz'
        with gzip.open(filename, "rt") as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.names = [row[0] for row in rows]
        self.countries = [row[1] for row in rows]
        self.len = len(self.countries)

        self.country_list = list(sorted(set(self.countries)))

    def __getitem__(self, index):
        return self.names[index], self.countries[index]

    def __len__(self):
        return self.len

    def get_countries(self):
        return self.country_list

    def get_country(self, id):
        return self.country_list[id]

    def get_country_id(self, country):
        return self.country_list.index(country)

# Test the loader
if __name__ == "__main__":
    dataset = NameDataset(False)
    print(dataset.get_countries())
    print(dataset.get_country(3))
    print(dataset.get_country_id('Korean'))

    train_loader = DataLoader(dataset=dataset,
                              batch_size=10,
                              shuffle=True)

    print(len(train_loader.dataset))
    for epoch in range(2):
        for i, (names, countries) in enumerate(train_loader):
            # Run your training process
            print(epoch, i, "names", names, "countries", countries)

