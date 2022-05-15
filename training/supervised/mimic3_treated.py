import torch
from torch.utils.data import Dataset
import sys
import os

from typing import Any, Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np
from torchvision.transforms import transforms

PATH = 'Users/ivo00/Desktop/Ivo/DTU/Courses/Semester 2/Special Course'

mimic3_dir = f"C:/{PATH}/mimic3-benchmarks-master"
sys.path.append(mimic3_dir)

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils


def read_and_extract_features(reader):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    period, features = 'all', 'all'

    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return X, ret['y'], ret['name']


def get_cached_filename(input: bool, folder: str) -> str:
    cached_dir = "datasets/cached"
    if input:
        filename = os.path.join(cached_dir, f"{folder}_X_cached.pt")
    else:
        filename = os.path.join(cached_dir, f"{folder}_y_cached.pt")
    return filename

        
def cached_data_exists(folder: str) -> bool:
    X_cached = get_cached_filename(True, folder)
    y_cached = get_cached_filename(False, folder)
    return os.path.exists(X_cached) and os.path.exists(y_cached)


def get_cached_data(folder: str):
    X_cached = get_cached_filename(True, folder)
    y_cached = get_cached_filename(False, folder)
    X = torch.load(X_cached)
    y = torch.load(y_cached)
    return (X, y)


class MIMIC3(Dataset):

    def __init__(
            self,
            train: bool = True,
            download: bool = True,
            use_cached: bool = True,
    ) -> None:
        self.transform = transforms.Compose([transforms.ToTensor()])
        folder = 'train' if train else 'test'

        # Check if we should use cached data
        if use_cached and cached_data_exists(folder):
            X, y = get_cached_data(folder)
        else:
            X, y = self.prepare_data(folder, download)

        # small fix - make sure the data always contains an even number of data points
        if len(X) % 2 > 0:
            X = X[:-1]
            y = y[:-1]

        self.data = (X, y)



    def prepare_data(self, folder: str, download: bool):
        reader = InHospitalMortalityReader(dataset_dir=mimic3_dir + f"/data/in-hospital-mortality/{folder}",
                                                listfile=mimic3_dir + f"/data/in-hospital-mortality/{folder}/listfile.csv",
                                                period_length=48.0)

        (X, y, names) = read_and_extract_features(reader)

        # Replace NaN values with the mean of that feature
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=True)
        imputer.fit(X)

        X = np.array(imputer.transform(X), dtype=np.float32)

        # Normalize data to have zero mean and unit variance
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        # Convert to Tensor
        X = torch.tensor(X).to(torch.float32)
        y = torch.tensor(y).to(torch.float32)
        y = torch.reshape(y, (len(y), 1))
        
        # Save the data
        if download:
            cached_dir = "datasets/cached"
            if not os.path.exists(cached_dir):
                os.makedirs(cached_dir)
            X_cached = os.path.join(cached_dir, f'{folder}_X_cached.pt')
            y_cached = os.path.join(cached_dir, f'{folder}_y_cached.pt')
            torch.save(X, X_cached)
            torch.save(y, y_cached)

        return (X, y)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.data[0][index]
        label = self.data[1][index]
        return data, label


    def __len__(self) -> int:
        return len(self.data[1])