from dataclasses import dataclass
from datetime import datetime
import random

from typing import List, Set

import torch

from model_cast import ModelTypedState
from nodes import Node
from utils import timing


@dataclass
class Job:
    ft_id: int
    deadline: datetime
    round_num: int
    processing_time_coef: float  # coefficient proportional to time required for aggregation
    model_states: List[ModelTypedState]
    # size_weights: List[float]

    @property
    def id(self):
        return self.ft_id * 10 ** 6 + self.round_num

    def __hash__(self):
        return self.id

    # def __post_init__(self):
    #     if len(self.model_states) != len(self.size_weights):
    #         raise ValueError(f"size_weights should be the same length as model_states")


class SFAggregationStationScheduler:
    # @timing
    @staticmethod
    def plan_next(jobs: Set[Job]):
        if not jobs:
            raise ValueError("No jobs provided")
        for j in jobs:
            j.reserve_coef = (j.deadline - datetime.now()) / (
                j.processing_time_coef)  # TODO check how it is calculated. Is it mock?
        min_reserve_coef = min(j.reserve_coef for j in jobs)
        best_candidates = [j for j in jobs if j.reserve_coef == min_reserve_coef]
        if len(best_candidates) == 1:
            return best_candidates[0]
        return random.choice(best_candidates)
        # TODO if best_candidates is short enough then
        #  find which one has best metric value (sum delay of deadline) with brute force


class RandomAggregationStationScheduler:
    @staticmethod
    def plan_next(jobs: Set[Job]):
        if not jobs:
            raise ValueError("No jobs provided")
        return random.choice(list(jobs))
