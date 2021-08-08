import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np

class TigerDataset(Dataset):

    def __init__(self, df):
        self.bags, self.labels = self.form_to_bag(df)

    def form_to_bag(self, df):
        # data_loader = pd.read_csv(csv_file)
        bags = []
        bag_labels = []
        for bag_num in df.bagName.unique():
            bag_ = df[df['bagName']==bag_num]
            instance = bag_.drop(['bag','bagName'], axis=1)
            label = bag_['bag'].iloc[0]
            bags.append(torch.tensor(instance.values))
            bag_labels.append(torch.tensor(label))
        return bags, bag_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = self.labels[idx]
        return bag, label, ''

if __name__ == '__main__':
    csv_file = 'tiger_inst.csv'
    df = pd.read_csv(csv_file)
    train_sample_size = 150
    dataloader = TigerDataset(df)
    batch_size=1
    training_sample = random.sample(range(len(dataloader)), train_sample_size)
    testing_sample = np.setdiff1d(range(len(dataloader)),training_sample)
    train = [dataloader[i] for i in training_sample]
    test = [dataloader[i] for i in testing_sample]
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)

