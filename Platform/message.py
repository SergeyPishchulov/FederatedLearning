from dataclasses import dataclass
from datetime import datetime
import torch


@dataclass
class MessageToHub:
    """Message from client to hub with information about updated model

    iteration_num - the serial number of the message sent.
    It can differ from the round at hub
    """
    iteration_num: int
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
