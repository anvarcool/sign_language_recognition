from collect_data import collect_data, filenames_train, filenames_test, file_labels_train, file_labels_test
from create_datasets import MyDataset, MyDataset_no_augment
from modeling import LSTMClassification
import torch
n_epochs = int(input('choose n_epochs'))
json_path = input('write the json path or "NO" if data is already tensor')
if json_path != "NO":
    data_path = input('write the path to store result of converting json to the tensors dataset:')
    collect_data(filenames_train, filenames_test, json_path, data_path)
else:
    data_path = input('write the path to the tensors dataset:')
model = LSTMClassification(128)
train = MyDataset(filenames_train, file_labels_train, True, data_path)
trainloader = torch.utils.data.DataLoader(train)
test = MyDataset_no_augment(filenames_test, file_labels_test, False, data_path)
testloader = torch.utils.data.DataLoader(test)
train_no_augment = MyDataset_no_augment(filenames_train, file_labels_train, True, data_path)
trainloader_no_augment = torch.utils.data.DataLoader(train_no_augment)
model.fit(trainloader, testloader, trainloader_no_augment, n_epochs, 0.4)
result = model.inference(testloader)