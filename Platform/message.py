from dataclasses import dataclass
from datetime import datetime, timedelta
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
    update_quality: float


@dataclass
class MessageToClient:
    round_num: int
    ft_id: int
    agr_model: torch.nn.Module
    should_finish: bool


@dataclass
class ResponseToHub:
    """
    Response to Hub on MessageToClient.

    It notifies the hub that aggregated model was received by client.
    delay is positive means the aggregated model was received after the required deadline. The hub will have penalty.
    delay is zero means the aggregated model was received before the required deadline.
    """
    client_id: int
    ft_id: int
    round_num: int
    delay: timedelta
