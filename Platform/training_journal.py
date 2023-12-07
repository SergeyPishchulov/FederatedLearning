from dataclasses import dataclass, astuple

import numpy as np
import torch
from datetime import datetime, date


@dataclass
class JournalRecord:
    model: torch.nn.Module
    deadline: datetime


class TrainingJournal:
    def __init__(self, task_ids):
        self.d = {}  # key is (ft_id, client_id, round); value is model
        self.latest_aggregated_round = {i: -1 for i in task_ids}

    def mark_as_aggregated(self, ft_id):
        self.latest_aggregated_round[ft_id] += 1
        # TODO bug if we skip some rounds

    def save_local(self, ft_id, client_id, round_num, model, deadline):
        if (ft_id, client_id, round_num) not in self.d:
            self.d[(ft_id, client_id, round_num)] = JournalRecord(model, deadline)
        else:
            raise KeyError("Key already exists")

    def _get_ft_ready_to_agr(self, client_ids):
        res = []
        for ft_id, latest_round in self.latest_aggregated_round.items():
            # print(f'Searching ({ft_id},_,{latest_round+1}) in keys. client_ids is {client_ids}')
            if all((ft_id, cl_id, latest_round + 1) in self.d
                   for cl_id in client_ids):  # TODO summ accuaracy > A
                # ft_records = [self.d[(ft_id, cl_id, latest_round + 1)] for cl_id in client_ids]
                res.append((ft_id, latest_round + 1))
        return res

    def get_ft_to_aggregate(self, client_ids):
        ready = self._get_ft_ready_to_agr(client_ids)
        if not ready:
            return (None,) * 4
        res = (datetime(year=3000), None, None, None)
        for ft_id, round in ready:
            records = [self.d[(ft_id, cl_id, round)] for cl_id in client_ids]
            models = [r.model for r in records]
            min_d = min([r.deadline for r in records])
            if min_d < res[0]:
                res = min_d, ft_id, round, models
        print(f'Task {res[1]} with min deadline {res[0]}')
        return res

    # print(f"No task to aggregate. d keys: {self.d.keys()}")
