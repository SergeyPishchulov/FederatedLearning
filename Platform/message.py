from dataclasses import dataclass
from datetime import datetime
import torch


@dataclass
class MessageToHub:
    round_num: int
    ft_id: int
    acc: float
    loss: float
    model: torch.nn.Module
    client_id: int
    deadline: datetime


@dataclass
class MessageToClient:
    round_num: int
    ft_id: int
    agr_model: torch.nn.Module
