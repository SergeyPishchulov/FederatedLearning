from datasets import Data
from nodes import Node


class FederatedMLTask:
    def __init__(self, id, args):
        self.args = args
        self.data = Data(args)  # TODO dumb
        print(f'DATA SIZE for task {id} {len(self.data.train_set)}')
        self.node_cnt = args.node_num
        self.id = id
        self.name = f'Task {id} {args.dataset} {args.local_model}'
        self.latest_agg_round = -1
        self.done = False
        # Data-size-based aggregation weights
        sample_size = []
        for i in range(self.node_cnt):
            sample_size.append(len(self.data.train_loader[i]))
        self.size_weights = [x / sum(sample_size) for x in sample_size]
        self.central_node = Node(-1, self.data.test_loader[0], self.data.test_set, self.args)
