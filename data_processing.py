import os
import re
import numpy as np
from tqdm import tqdm

def dataProcessing(file_path, use_sliding_window=True, step_size=5120):
    train_filenames = os.listdir(file_path)
    def capture(path, filenames, start_row=0, end_row=None):
        pressure_data, vibration_data = {}, {}
        for filename in tqdm(filenames):
            file_path = os.path.join(path, filename)
            max_rows = None
            if end_row is not None:
                max_rows = end_row - start_row
            try:
                file = np.genfromtxt(file_path, delimiter=",", dtype=float,skip_header=start_row, max_rows=max_rows)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            pressure = file[:, 0:1]
            vibration = file[:, 1:]

            pressure_data[filename] = pressure
            vibration_data[filename] = vibration
        return pressure_data, vibration_data

    def slice(pressure_data, vibration_data):
        Data_P, Data_V, Test_P, Test_V = [], [], [], []
        Labels, Test_Labels = [], []

        for key in tqdm(pressure_data.keys()):
            p = pressure_data[key]
            v = vibration_data[key]

            match = re.search(r'电流-([A-Z+]+)-S', key)
            key_part = match.group(1) if match else 'UNKNOWN'
            key_mapping = {
                'A': 0,
                'C': 1,
                'G': 2,
                'N': 3,
                'B': 4,
                'B+G': 5,
                'H': 6,
                'H+N': 7,
                'E': 8,
            }
            class_num = key_mapping.get(key_part, -1)

            length = p.shape[0]
            num_segments = length // step_size
            start = 0

            for i in range(num_segments):
                p_seg = p[start:start + step_size]
                v_seg = v[start:start + step_size]

                if p_seg.shape[0] == step_size and v_seg.shape[0] == step_size:
                    if use_sliding_window:
                        if i >= round(0.8 * num_segments):
                            Test_P.append(p_seg)
                            Test_V.append(v_seg)
                            Test_Labels.append(class_num)
                        else:
                            Data_P.append(p_seg)
                            Data_V.append(v_seg)
                            Labels.append(class_num)
                start += step_size

        return (
            np.array(Data_P), np.array(Data_V), np.array(Labels),
            np.array(Test_P), np.array(Test_V), np.array(Test_Labels)
        )

    pressure_data, vibration_data = capture(file_path, train_filenames, start_row=0, end_row=614400)
    Train_P, Train_V, Train_Y, Test_P, Test_V, Test_Y = slice(pressure_data, vibration_data)

    return Train_P, Train_V, Train_Y, Test_P, Test_V, Test_Y

def standardize_p(Train_X, Test_X):
    mean = np.mean(Train_X, axis=(0, 1), keepdims=True)
    std = np.std(Train_X, axis=(0, 1), keepdims=True)
    Train_X = (Train_X - mean) / std
    Test_X = (Test_X - mean) / std
    return Train_X, Test_X


def standardize_with_train(train_data, test_data):

    s = np.array([0,0,0])
    s1 = np.broadcast_to(s, (5120,3))
    valid_train_mask = np.stack([np.all(item == s1, axis=(0, 1)) for item in train_data])
    valid_train_mask = ~valid_train_mask
    valid_train = train_data[valid_train_mask]

    mean = valid_train.mean(axis=(0, 1), keepdims=True)
    std = valid_train.std(axis=(0, 1), keepdims=True)
    print(mean,std)

    def standardize(data):
        data_std = data.copy()
        valid_mask = np.stack([np.all(item == s1, axis=(0, 1)) for item in data])
        valid_mask = ~valid_mask
        data_std[valid_mask] = (data[valid_mask] - mean) / std
        return data_std


    train_data_std = standardize(train_data)
    test_data_std = standardize(test_data)

    return train_data_std, test_data_std

def collect_data1(batch):

    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    y = np.array(labels)
    data = np.array(data)

    S_P = data[:, :, 0:1]
    S_V = data[:, :, 1:4]
    S_P1= data[:, :, 4:5]

    s = np.array([0,0,0])
    s1 = np.broadcast_to(s, (5120, 3))

    is_zero = np.stack([np.all(item == s1, axis=(0,1)) for item in S_V])  # shape: [48]

    pairs = ~is_zero

    return [y, S_P, S_V, pairs,S_P1]



