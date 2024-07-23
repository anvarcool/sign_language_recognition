from collect_data import collect_data
from create_datasets import MyDataset, MyDataset_no_augment
from modeling import LSTMClassification
import torch
import pandas as pd
import numpy as np
data_path = input('Print path to data:')
n_epochs = int(input('Print n_epochs'))
json_path = input('print json path or "NO" if data is already tensor')
annotations = pd.read_csv(f'{data_path}/annotations.csv', sep = '\t')
annotations_train = annotations.query('train & (text != "no_event")')
annotations_test = annotations.query('~train & (text != "no_event")')
labels_map = {name : i for
              i, name in enumerate(annotations_train['text'].unique())}
filenames_train = np.array(annotations_train['attachment_id'])
file_labels_train = np.array([labels_map[key] for key in annotations_train['text']])
filenames_test = np.array(annotations_test['attachment_id'])
file_labels_test = np.array([labels_map[key] for key in annotations_test['text']])
if json_path != "NO":
    collect_data(filenames_train, filenames_test, json_path, data_path)
model = LSTMClassification(128)
train = MyDataset(filenames_train, file_labels_train, True, data_path)
trainloader = torch.utils.data.DataLoader(train)
test = MyDataset_no_augment(filenames_test, file_labels_test, False, data_path)
testloader = torch.utils.data.DataLoader(test)
train_no_augment = MyDataset_no_augment(filenames_train, file_labels_train, True, data_path)
trainloader_no_augment = torch.utils.data.DataLoader(train_no_augment)
model.fit(trainloader, testloader, trainloader_no_augment, n_epochs, 0.4)
result = model.inference(testloader)