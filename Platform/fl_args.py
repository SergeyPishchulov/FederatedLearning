from typing import List

from datasets import Data


class FLConfig:
    def __init__(self, node_cnt, aggr_algorithm):
        self.node_cnt = node_cnt
        self.aggr_algorithm = aggr_algorithm# ex server_method
