import torch
from torch.utils.data import Dataset
import sys
import os
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from torchvision.transforms import transforms

PATH = 'Users/ivo00/Desktop/Ivo/DTU/Courses/Semester 2/Special Course'

home_dir = f"C:/{PATH}/mimic3-benchmarks-master"
sys.path.append(home_dir)

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models.in_hospital_mortality import utils
from mimic3models import common_utils


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

        discretizer = Discretizer(timestep=1.0,
                                  store_masks=True,
                                  impute_strategy='previous',
                                  start_time='zero')

        discretizer_header = discretizer.transform(reader.read_example(0)["X"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
        normalizer_state = None
        if normalizer_state is None:
           normalizer_state = 'ihm_ts1.0.input_str_previous.start_time_zero.normalizer'
           normalizer_state = home_dir + f"/mimic3models/in_hospital_mortality/{normalizer_state}"
        normalizer.load_params(normalizer_state)

        load_partial_data = False
        self.data = utils.load_data(reader, discretizer, normalizer, load_partial_data)

        trimmed_data = torch.tensor(self.data[0][:, 0, :]).to(torch.float32)

        transformed_labels = torch.tensor(self.data[1]).to(torch.float32)
        transformed_labels = torch.reshape(transformed_labels, (len(transformed_labels), 1))

        self.data = (trimmed_data, transformed_labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.data[0][index]
        label = self.data[1][index]
        return data, label

    def __len__(self) -> int:
        return len(self.data[1])
