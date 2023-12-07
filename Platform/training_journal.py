from dataclasses import dataclass, astuple
import torch
from datetime import datetime


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

    def get_ft_to_aggregate(self, client_ids):
        for ft_id, latest_round in self.latest_aggregated_round.items():
            # print(f'Searching ({ft_id},_,{latest_round+1}) in keys. client_ids is {client_ids}')
            if all((ft_id, cl_id, latest_round + 1) in self.d
                   for cl_id in client_ids):
                records = [astuple(self.d[(ft_id, cl_id, latest_round + 1)]) for cl_id in client_ids]
                models, deadlines = zip(*records)
                return ft_id, latest_round + 1, models, deadlines
        # print(f"No task to aggregate. d keys: {self.d.keys()}")
        return None, None, None, None
