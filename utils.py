import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def evaluate_RNN(model, test_data):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in (pbar := tqdm(test_data)):
            inputs = inputs.flatten(start_dim = 1, end_dim = 2).permute((0, 2, 1)).to(device)
            labels = labels.to(device)
            accuracy += torch.sum(torch.argmax(model(inputs), dim = 1) == labels.to(device))
            pbar.set_description(f'num_of_right_preds : {accuracy} ')
    model.train()
    return accuracy / (len(test_data) * test_data.batch_size)