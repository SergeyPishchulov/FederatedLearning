from dataclasses import dataclass
from datetime import datetime
import random

from typing import List

import torch

from utils import timing


@dataclass
class Job:
    ft_id: int
    deadline: datetime
    round_num: int
    processing_time_coef: float  # coefficient proportional to time required for aggregation
    models: List[torch.Module]


class SFAggregationStationScheduler:
    # @timing
    @staticmethod
    def plan_next(jobs: List[Job]):
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
    def plan_next(jobs: List[Job]):
        if not jobs:
            raise ValueError("No jobs provided")
        return random.choice(jobs)
