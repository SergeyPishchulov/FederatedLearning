from federated_ml_task import FederatedMLTask
from nodes import ClientMLTask
from typing import List


class Client:
    def __init__(self, num: int):
        self.num = num
        self.tasks: List[ClientMLTask] = []


