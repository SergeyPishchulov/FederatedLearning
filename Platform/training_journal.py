import argparse
from dataclasses import dataclass, astuple

import numpy as np
import torch
from datetime import datetime, date

from typing import Dict


@dataclass
class JournalRecord:
    """
    Representation of a client's message.

    model - locally trained model
    deadline - max time the aggregation should have been done to
    """
    model: torch.nn.Module
    deadline: datetime
    update_quality: float


class TrainingJournal:
    def __init__(self, task_ids, required_quality, args):  # TODO should not have args
        self.d: Dict[tuple, JournalRecord] = {}  # key is (ft_id, client_id, round); value is Record(model, deadline)
        self.latest_aggregated_round = {i: -1 for i in task_ids}
        self.required_quality_by_ft_id = required_quality
        self.args = args

    def mark_as_aggregated(self, ft_id):
        self.latest_aggregated_round[ft_id] += 1
        # TODO bug if we skip some rounds

    def save_local(self, ft_id, client_id, round_num, model, deadline, update_quality):
        if (ft_id, client_id, round_num) not in self.d:
            self.d[(ft_id, client_id, round_num)] = JournalRecord(model, deadline, update_quality)
        else:
            raise KeyError("Key already exists")

    def _get_ft_ready_to_agr(self, client_ids):
        res = []
        if self.args.aggregation_on == 'all_received':
            decision_func = self.all_clients_performed_round
        elif self.args.aggregation_on == 'required_quality':
            decision_func = self.required_quality_reached
        else:
            raise argparse.ArgumentError(self.args.aggregation_on, "Unknown value")
        for ft_id, latest_round in self.latest_aggregated_round.items():
            condition = decision_func(ft_id, latest_round, client_ids)
            if condition:
                res.append((ft_id, latest_round + 1))
        return res

    def all_clients_performed_round(self, ft_id, round_num, client_ids):
        return all((ft_id, cl_id, round_num + 1) in self.d
                   for cl_id in client_ids)

    def required_quality_reached(self, ft_id, round_num, client_ids):
        sum_quality = 0
        for cl_id in client_ids:
            if (ft_id, cl_id, round_num + 1) in self.d:
                sum_quality += self.d[(ft_id, cl_id, round_num + 1)].update_quality
        print(
            f"JOURNAL: Quality reached/required = {round(sum_quality / self.required_quality_by_ft_id[ft_id], 3)}. SUM is{sum_quality}")
        if sum_quality >= self.required_quality_by_ft_id[ft_id]:
            print(f"JOURNAL: Quality reached.")
            return True
        # print(f"JOURNAL: Quality reached/required = {round(sum_quality / self.required_quality_by_ft_id[ft_id], 3)}")
        return False or self.all_clients_performed_round(ft_id, round_num, client_ids)

    def get_ft_to_aggregate(self, client_ids):
        ready = self._get_ft_ready_to_agr(client_ids)
        if not ready:
            return {}
        total_min_deadline = datetime.max
        res_tasks = {}
        for ft_id, round_num in ready:
            # TODO think what will happen dith updates that was sent after aggreagation
            # NOTE: minimal deadline of task is computed only by clients who have sent an update
            # Fair enough: clients who havent sent can not vote for aggregation of the task
            records = []
            for cl_id in client_ids:
                if (ft_id, cl_id, round_num) in self.d:
                    records.append(self.d[(ft_id, cl_id, round_num)])
            models = [r.model for r in records]
            min_d = min([r.deadline for r in records])  # feature of the task
            res_tasks[ft_id] = ((min_d, round_num, models))
            if min_d < total_min_deadline:
                total_min_deadline = min_d
        # print(f'Task {res[1]} with min deadline {res[0]}')
        # return res
        return res_tasks

    # print(f"No task to aggregate. d keys: {self.d.keys()}")
