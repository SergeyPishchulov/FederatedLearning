from dataclasses import dataclass


@dataclass
class MessageToHub:
    round: int
    ft_id: int
    acc: float
    loss: float
    model: object


@dataclass
class MessageToClient:
    round: int
    ft_id: int
    agr_model: object
