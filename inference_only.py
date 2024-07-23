from modeling import LSTMClassification
from utils import evaluate_RNN
from create_datasets import MyDataset_no_augment
import numpy as np
import pandas as pd
import torch
models_path = input('Print path to data:')
data_path = input('Print path to data:')
annotations = pd.read_csv(f'{data_path}/annotations.csv', sep = '\t')
annotations_train = annotations.query('train & (text != "no_event")')
annotations_test = annotations.query('~train & (text != "no_event")')
labels_map = {name : i for
              i, name in enumerate(annotations_train['text'].unique())}
filenames_train = np.array(annotations_train['attachment_id'])
file_labels_train = np.array([labels_map[key] for key in annotations_train['text']])
filenames_test = np.array(annotations_test['attachment_id'])
file_labels_test = np.array([labels_map[key] for key in annotations_test['text']])
model = LSTMClassification(128, models_path)
test = MyDataset_no_augment(filenames_test, file_labels_test, False, data_path)
testloader = torch.utils.data.DataLoader(test)
model.load_state_dict(torch.load(f'{models_path}/final_model_weights.pth'))
evaluate_RNN(model, testloader)

predicts = model.inference(testloader)
