from datasets import Data, DataLoader


class FederatedMLTaskConfiguration:
    def __init__(self, dataset, nn_architecture, epochs):
        self.dataset = dataset
        self.nn_architecture = nn_architecture
        self.epochs = epochs
        self.iid = 1  # data is iid


class FederatedMLTask:
    def __init__(self, node_cnt, conf: FederatedMLTaskConfiguration):
        self.data = DataLoader(node_cnt, conf.dataset, conf)  # TODO dumb
        self.conf = conf
        self.node_cnt = node_cnt
        # Data-size-based aggregation weights
        sample_size = []
        for i in range(node_cnt):
            sample_size.append(len(self.data.train_loader[i]))
        self.size_weights = [x / sum(sample_size) for x in sample_size]
        self.central_node = None
        self.client_nodes = None
