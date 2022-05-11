import torch
from torch.utils.data import Dataset
import sys

from typing import Any, Callable, Optional, Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np
from torchvision.transforms import transforms

PATH = 'Users/ivo00/Desktop/Ivo/DTU/Courses/Semester 2/Special Course'

home_dir = f"C:/{PATH}/mimic3-benchmarks-master"
sys.path.append(home_dir)

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models.in_hospital_mortality import utils
from mimic3models import common_utils


def read_and_extract_features(reader):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    period, features = 'all', 'all'

    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return X, ret['y'], ret['name']


class MIMIC3(Dataset):

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        folder = 'train' if train else 'test'
        self.transform = transforms.Compose([transforms.ToTensor()])
        reader = InHospitalMortalityReader(dataset_dir=home_dir + f"/mimic3/{folder}",
                                                listfile=home_dir + f"/mimic3/{folder}/listfile.csv",
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

        self.data = (X, y)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.data[0][index]
        label = self.data[1][index]
        return data, label

    def __len__(self) -> int:
        return len(self.data[1])

