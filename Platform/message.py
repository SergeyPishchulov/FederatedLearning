from dataclasses import dataclass

import torch


@dataclass
class MessageToHub:
    round: int
    ft_id: int
    acc: float
    loss: float
    model: torch.nn.Module
    client_id: int


@dataclass
class MessageToClient:
    round: int
    ft_id: int
    agr_model: torch.nn.Module
