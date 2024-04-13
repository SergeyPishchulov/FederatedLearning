from dataclasses import dataclass
from datetime import datetime, timedelta
import torch
from typing import Optional, Dict, List, Tuple

from model_cast import ModelTypedState
from aggregation_station import Job
from nodes import Node
from utils import norm


@dataclass
class Period:
    start: datetime
    end: datetime

    def norm(self, global_start_time: datetime):
        return Period(norm(self.start, global_start_time),
                      norm(self.end, global_start_time))

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
    model_state: ModelTypedState
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
    should_run: bool


@dataclass
class MessageAgsToClient:
    ft_id: int
    round_num: int
    agr_model_state: ModelTypedState


@dataclass
class MessageAgsToHub:
    ft_id: int
    round_num: int
    agr_model_state: ModelTypedState
    period: Period
    jobs_cnt_in_time: List[Tuple[datetime, int]]


@dataclass
class ControlMessageToClient:
    should_run: bool
    start_time: Optional[datetime]


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
    # final_message: bool = False


@dataclass
class MessageToValidator:
    ft_id: int
    ag_round_num: int
    model_state: Optional[ModelTypedState]
    should_finish: bool = False


@dataclass
class ValidatorShouldFinish:
    pass


@dataclass
class ControlValidatorMessage:
    start_time: datetime


@dataclass
class MessageValidatorToHub:
    ft_id: int
    ag_round_num: int
    acc: float


FT_ID = int


@dataclass
class MessageHubToAGS:
    jobs_by_ft_id: Dict[FT_ID, Job]


@dataclass
class ControlMessageHubToAGS:
    start_time: datetime


@dataclass
class FinishMessageToAGS:
    pass
