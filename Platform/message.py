from dataclasses import dataclass
from datetime import datetime, timedelta
import torch
from typing import Optional

from nodes import Node


@dataclass
class Period:
    start: datetime
    end: datetime

    def __str__(self):
        return f'[{self.start.strftime("%H:%M:%S")} --- {self.end.strftime("%H:%M:%S")}]\n'

    def __repr__(self):
        return self.__str__()


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
    round_num: int  # In which round client want to participate.
    # model is fine-tuned model from prev round. See MessageToClient
    period: Period
    time_to_target_acc_sec: int  # TODO delete


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
    final_message: bool = False


@dataclass
class MessageToValidator:
    node: Optional[Node]
    should_finish: bool = False


@dataclass
class MessageValidatorToHub:
    acc: float
