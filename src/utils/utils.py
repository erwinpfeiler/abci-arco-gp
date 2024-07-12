import sys
from typing import Dict

import pandas as pd
import torch
from torch.nn.utils import parameters_to_vector


def inf_tensor():
    return torch.tensor(float('Inf'))


def get_module_params(m):
    return parameters_to_vector(m.parameters()).detach().cpu()


def sizeof(obj, scale=1e-6, print_threshold=-1.):
    size = sys.getsizeof(obj) * scale
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += sys.getsizeof(k) * scale
            tmp = sizeof(v, scale, print_threshold)
            size += tmp
            if 0. < print_threshold <= tmp:
                print(k, tmp)
    elif isinstance(obj, torch.Tensor):
        size += sys.getsizeof(obj.untyped_storage()) * scale

    return size


def export_stats(stats: Dict[str, list], path: str):
    data = []
    columns = []
    for key, value_list in stats.items():
        if len(value_list):
            columns.append(key)
            data.append(pd.DataFrame(torch.tensor(value_list).view(-1)))

    if len(columns):
        df = pd.concat(data, axis=1)
        df.columns = columns
        df.to_csv(path, index=False)
