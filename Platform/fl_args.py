from typing import List


class LocalArgs:
    def __init__(self, dataset, nn_architecture, epochs):
        self.dataset = dataset
        self.nn_architecture = nn_architecture
        self.epochs = epochs


class FLArgs:
    def __init__(self, node_cnt, aggr_algorithm, clients_args: List[LocalArgs]):
        self.node_cnt = node_cnt
        self.aggr_algorithm = aggr_algorithm
        self.clients_args = clients_args
