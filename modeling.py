from utils import evaluate_RNN
from torch import nn
import torch
from tqdm import tqdm
import gc
from torch import optim
import time
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LSTMClassification(nn.Module):
    def __init__(self, hidden_dim, pretrained_weights_path):
        super(LSTMClassification, self).__init__(hidden_dim, pretrained_weights_path)
        self.conv1 = nn.Sequential(
        nn.Conv1d(128, 256, kernel_size = 1, groups = 8),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace = True),
        nn.TransformerEncoderLayer(d_model=126, nhead = 3, batch_first=True),
        nn.Conv1d(256, 256, kernel_size = 3, groups = 64),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace = True),
        nn.Conv1d(256, 128, kernel_size = 1, groups = 8),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace = True),
        nn.TransformerEncoderLayer(d_model=124, nhead = 1, batch_first=True),
        )
        self.conv2 = nn.Sequential(
        nn.Conv1d(128, 256, kernel_size = 1, groups = 8),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace = True),
        nn.TransformerEncoderLayer(d_model= 124, nhead = 1, batch_first=True, dropout = 0.1),
        nn.Conv1d(256, 256, kernel_size = 3, groups = 64),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace = True),
        nn.Conv1d(256, 128, kernel_size = 1, groups = 8),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace = True),
        nn.TransformerEncoderLayer(d_model= 122, nhead = 1, batch_first=True),
        )
        self.lstm = nn.LSTM(122, hidden_dim,num_layers = 3, batch_first=True, dropout=0.3)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 250)
        self.to(device)
        self.load_state_dict(torch.load(f'{self.pretrained_weights_path}/pretrained_on_asl_model_weights.pth'))
        self.conv3 = nn.Sequential(
                nn.Conv1d(128, 256, kernel_size = 1, groups = 8),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace = True),
                nn.TransformerEncoderLayer(d_model= 122, nhead = 1, batch_first=True, dropout = 0.1),
                nn.Conv1d(256, 256, kernel_size = 3, groups = 64),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace = True),
                nn.Conv1d(256, 128, kernel_size = 1, groups = 8),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace = True),
                nn.TransformerEncoderLayer(d_model= 120, nhead = 3, batch_first=True, dropout = 0.1),
                )
        self.lstm = nn.LSTM(120, 128,num_layers = 2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(128, 500),
            nn.ReLU(inplace = True),
            nn.Linear(500, 1000)
        )
    def forward(self, input_):
        x = self.conv1(input_)
        x = self.conv2(x)
        x = self.conv3(x)
        lstm_out, (h, c) = self.lstm(x)
        x = self.norm(lstm_out[:, -1, :])
        logits = self.fc(x)
        return logits
    def fit(self, trainloader, testloader, trainloader_no_augment, n_epochs, min_accuracy):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr = 0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.9)
        self.to(device)
        acc_train = []
        acc_test = []
        for epoch in range(n_epochs):
            running_loss = 0
            start_time = time.time()
            for inputs, labels in (pbar := tqdm(trainloader)):
                inputs = inputs.flatten(start_dim = 1, end_dim = 2).permute((0, 2, 1)).to(device)
                labels = labels.to(device)
                outputs = self.forward(inputs)
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                running_loss += loss_item
                pbar.set_description(f'loss : {loss_item} ')
            end_time = time.time()
            scheduler.step()
            print('epoch: ', epoch, 'loss: ', running_loss / (15000 / trainloader.batch_size), 'time: ', end_time - start_time)
            if (epoch + 1) % 20 == 0:
                accuracy_test = evaluate_RNN(self, testloader)
                accuracy_train = evaluate_RNN(self, trainloader_no_augment)
                acc_train.append(accuracy_train.item())
                acc_test.append(accuracy_test.item())
                print(accuracy_test, accuracy_train)
                gc.collect()
                torch.cuda.empty_cache()
                if accuracy_test > min_accuracy:
                    torch.save(self.state_dict(), f'rsl_model_weights_{epoch + 1}.pth')
    def inference(self, testloader):
        result = np.array([])
        for inputs, labels in testloader:
            inputs = inputs.flatten(start_dim = 1, end_dim = 2).permute((0, 2, 1)).to(device)
            outputs = self.forward(inputs)
            result = np.append(result, torch.argmax(outputs, dim = 1).numpy())
        return result
