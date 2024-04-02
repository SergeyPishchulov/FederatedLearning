import time

from model_cast import ModelCast
from utils import validate
from message import MessageToValidator, MessageValidatorToHub, ValidatorShouldFinish, ControlValidatorMessage
import torch
import os
from utils import setup_seed
from datetime import datetime
from typing import Optional


class Validator:
    def __init__(self, user_args, node_by_ft_id):
        self.user_args = user_args
        self.node_by_ft_id = node_by_ft_id
        self.should_finish = False
        self.start_time: Optional[datetime] = None

    def handle_messages(self, read_q, write_q, ):
        while not read_q.empty():
            mes = read_q.get()
            if isinstance(mes, MessageToValidator):
                if self.start_time is None:
                    raise ValueError("Got MessageToValidator when validator is not started")
                if self.should_finish:
                    return
                node = self.node_by_ft_id[mes.ft_id]
                ModelCast.to_model(mes.model_state, node.model)
                acc = validate(args=None,
                               node=node,
                               which_dataset='local')
                response = MessageValidatorToHub(mes.ft_id,
                                                 mes.ag_round_num,
                                                 acc)
                write_q.put(response)
            elif isinstance(mes, ValidatorShouldFinish):
                self.should_finish = True
                # del mes# TODO do we need it here?
                return
            elif isinstance(mes, ControlValidatorMessage):
                self.start_time = mes.start_time
            else:
                raise ValueError(f"Unknown message {mes}")
            del mes

    def setup(self):
        setup_seed(self.user_args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.user_args.device
        torch.cuda.set_device('cuda:' + self.user_args.device)

    def run(self, read_q, write_q):
        while self.start_time is None:
            # print(f"Validator is waiting")
            self.handle_messages(read_q, write_q)
            time.sleep(1)
        if datetime.now() < self.start_time:
            delta = (self.start_time - datetime.now()).total_seconds()
            time.sleep(delta)
        self.setup()
        print("Validator started")
        while not self.should_finish:
            self.handle_messages(read_q, write_q)
        print("Validator stopped")
