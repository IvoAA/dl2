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
        self.reader = InHospitalMortalityReader(dataset_dir=home_dir + f"/mimic3/{folder}",
                                                listfile=home_dir + f"/mimic3/{folder}/listfile.csv",
                                                period_length=48.0)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        item = self.reader.read_example(index)
        data = item.get('X')
        label = item['y']
        return data, label

    def __len__(self) -> int:
        return self.reader.get_number_of_examples()

    def __iter__(self) -> int:
        return self.reader.read_next()

    # def _next_data(self):
    #     return self.reader.read_next()
