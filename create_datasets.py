import torch
import random
import numpy as np
import pandas as pd
annotations = pd.read_csv(f'annotations.csv', sep = '\t')
annotations_train = annotations.query('train & (text != "no_event")')
annotations_test = annotations.query('~train & (text != "no_event")')
labels_map = {name : i for
              i, name in enumerate(annotations_train['text'].unique())}
filenames_train = np.array(annotations_train['attachment_id'])
file_labels_train = np.array([labels_map[key] for key in annotations_train['text']])
filenames_test = np.array(annotations_test['attachment_id'])
file_labels_test = np.array([labels_map[key] for key in annotations_test['text']])
def affine(index, t, p):
    rand = random.uniform(0, 1)
    if rand < p:
        theta = random.uniform(-0.3, 0.3)
        z = torch.zeros(t.size())
        Q = torch.tensor(np.array([np.cos(theta), np.sin(theta)]))
        h = annotations_train.iloc[index, 3]
        w = annotations_train.iloc[index, 4]
        z[0], z[1] = (Q[0] * t[0]  - Q[1] * t[1] * h / w, 
                                         Q[1] * t[0] * w / h + Q[0] * t[1])
        b = [random.uniform(-0.5,0.5), random.uniform(-0.5,0.5), random.uniform(0,1)]
        b0 = min(1 - torch.max(z[0]), b[0]) if b[0] > 0 else max(-torch.min(z[0]), b[0])
        b1 = min(1 - torch.max(z[1]), b[1]) if b[1] > 0 else max(-torch.min(z[1]), b[1])
        bias = [b0, b1, b[2]]
        z[0], z[1], z[2] = z[0] + bias[0], z[1] + bias[1], z[2] + bias[2]
        return z
    else:
        return t
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, names, labels, is_train, data_path):
        super().__init__()
        self.names = names
        self.labels = labels
        filename = 'train' if is_train else 'test'
        self.data = torch.cat((torch.load(f'{data_path}/{filename}.pt'), 
                             torch.load(f'{data_path}/{filename}_h2.pt')), dim = 2)
    def __len__(self):
        return len(self.names)
    def __getitem__(self, index):
        p = random.uniform(0, 1)
        z = affine(index, self.data[index,:, :, :128])
        if p < 0.5:
            return torch.stack((1 - z[0], z[1], z[2]), dim = 0), self.labels[index]
        else:
            return z, self.labels[index]
class MyDataset_no_augment(torch.utils.data.Dataset):
    def __init__(self, names, labels, is_train, data_path):
        super().__init__()
        self.names = names
        self.labels = labels
        filename = 'train' if is_train else 'test'
        self.data = torch.cat((torch.load(f'{data_path}/{filename}_data.pt'), 
                             torch.load(f'{data_path}/{filename}_data_h2.pt')), dim = 2)
    def __len__(self):
        return len(self.names)
    def __getitem__(self, index):
        return self.data[index,:, :, :128], self.labels[index]

