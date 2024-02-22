from FederatedLearning.Platform.utils import validate
from message import MessageToValidator, MessageValidatorToHub, ValidatorShouldFinish


class Validator:
    def __init__(self):
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

    def run(self, read_q, write_q):
        while not self.should_finish:
            self.handle_messages(read_q, write_q)
