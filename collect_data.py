from tqdm import tqdm
import torch
import gc
import json

def parse_json(json_path):    
    with open(json_path, 'r') as f:
            data = f.read()
    jsons = {}
    ind = 1
    l = len(data)
    while ind < l:
        n = 1
        k = 0
        ch = ''
        string_json = ''
        while ch != '[':
            ch = data[ind]
            string_json += ch
            ind += 1
        username = string_json[1:-4]
        while (n != 0) or (k != 0):
            ch = data[ind]
            if ch == '{':
                k += 1
            if ch == '[':
                n += 1
            if ch == '}':
                k -= 1
            if ch == ']':
                n -=1
            string_json += ch
            ind += 1
        jsons[username] = '{' + string_json + '}'
        ind += 2
    return jsons


def collect_data(filenames_train, filenames_test, json_path, data_path):
    jsons = parse_json(json_path)
    #Train
    train_h2 = torch.zeros(len(filenames_train), 3, 21, 256)
    train = torch.zeros(len(filenames_train), 3, 21, 256)
    for num, item in enumerate(tqdm(filenames_train)):
        d = json.loads(jsons[item])[f'"{item}"']
        out = torch.zeros(3, 21, len(d))
        out_h2 = torch.zeros(3, 21, len(d))
        for i, frame in enumerate(d):
            if 'hand 2' in frame.keys():
                for j, landmark in enumerate(frame['hand 2']):
                    out_h2[0][j][i] = landmark['x']
                    out_h2[1][j][i] = landmark['y']
                    out_h2[2][j][i] = landmark['z']
            if 'hand 1' in frame.keys():
                for j, landmark in enumerate(frame['hand 1']):
                    out[0][j][i] = landmark['x']
                    out[1][j][i] = landmark['y']
                    out[2][j][i] = landmark['z']
        train_h2[num] = out_h2
        train[num] = out
        gc.collect()
        torch.save(train_h2, f'{data_path}/train_h2.pt')
        torch.save(train, f'{data_path}/train.pt')
    #Test
    test_h2 = torch.zeros(len(filenames_test), 3, 21, 256)
    test = torch.zeros(len(filenames_test), 3, 21, 256)
    for num, item in enumerate(tqdm(filenames_test)):
        d = json.loads(jsons[item])[f'"{item}"']
        out = torch.zeros(3, 21, len(d))
        out_h2 = torch.zeros(3, 21, len(d))
        for i, frame in enumerate(d):
            if 'hand 2' in frame.keys():
                for j, landmark in enumerate(frame['hand 2']):
                    out_h2[0][j][i] = landmark['x']
                    out_h2[1][j][i] = landmark['y']
                    out_h2[2][j][i] = landmark['z']
            if 'hand 1' in frame.keys():
                for j, landmark in enumerate(frame['hand 1']):
                    out[0][j][i] = landmark['x']
                    out[1][j][i] = landmark['y']
                    out[2][j][i] = landmark['z']
        test_h2[num] = out_h2
        test[num] = out
        gc.collect()
        torch.save(test_h2, f'{data_path}/test_h2.pt')
        torch.save(test, f'{data_path}/test.pt')
    gc.collect()