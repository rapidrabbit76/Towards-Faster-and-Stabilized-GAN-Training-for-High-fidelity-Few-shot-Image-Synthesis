import typing as T
from dataclasses import dataclass
from pydantic import DirectoryPath


@dataclass
class Hyperparameters:
    seed: int
    data_root: DirectoryPath
    start_iter: int
    total_iterations: int
    checkpoint: T.Optional[DirectoryPath]
    batch_size: int
    image_size: int
    ndf: int
    ngf: int
    nz: int
    lr: float
    beta: T.Any
    device: str
    num_workers: int
    interval: T.Any
    save_model_path: DirectoryPath
    policy: str
    multi_gpu: bool = True
