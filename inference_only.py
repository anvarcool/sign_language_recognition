from modeling import LSTMClassification
from utils import evaluate_RNN
from create_datasets import MyDataset_no_augment
from collect_data import collect_data, filenames_train, filenames_test, file_labels_test
import numpy as np
import pandas as pd
import torch
model = LSTMClassification(128)
json_path = input('write the json path or "NO" if data is already tensor')
if json_path != "NO":
    data_path = input('write the path to store result of converting json to the tensors dataset:')
    collect_data(filenames_train, filenames_test, json_path, data_path)
else:
    data_path = input('write the path to the tensors dataset:')
test = MyDataset_no_augment(filenames_test, file_labels_test, False, data_path)
testloader = torch.utils.data.DataLoader(test, batch_size = 40)
model.load_state_dict(torch.load('models/final_model_weights.pth'))
evaluate_RNN(model, testloader)

predicts = model.inference(testloader)
