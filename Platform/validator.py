from utils import validate
from message import MessageToValidator, MessageValidatorToHub, ValidatorShouldFinish
import torch
import os
from utils import setup_seed


class Validator:
    def __init__(self, user_args):
        self.user_args = user_args
        self.should_finish = False

    def handle_messages(self, read_q, write_q):
        while not read_q.empty():
            mes = read_q.get()
            if isinstance(mes, MessageToValidator):
                if self.should_finish:
                    return

                acc = validate(args=None,
                               node=mes.node,
                               which_dataset='local')
                response = MessageValidatorToHub(mes.ft_id,
                                                 mes.ag_round_num,
                                                 acc)
                write_q.put(response)
            elif isinstance(mes, ValidatorShouldFinish):
                self.should_finish = True
                # del mes# TODO do we need it here?
                return
            else:
                raise ValueError(f"Unknown message {mes}")
            del mes

    def setup(self):
        setup_seed(self.user_args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.user_args.device
        torch.cuda.set_device('cuda:' + self.user_args.device)

    def run(self, read_q, write_q):
        self.setup()
        while not self.should_finish:
            self.handle_messages(read_q, write_q)
