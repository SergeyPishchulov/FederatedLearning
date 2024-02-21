from FederatedLearning.Platform.utils import validate
from message import MessageToValidator, MessageValidatorToHub


class Validator:
    def __init__(self):
        self.should_finish = False

    def handle_messages(self, read_q, write_q):
        while not read_q.empty():
            mes: MessageToValidator = read_q.get()
            self.should_finish = mes.should_finish

            if self.should_finish:
                return

            acc = validate(args=None,
                           node=mes.node,
                           which_dataset='local')
            response = MessageValidatorToHub(acc)
            write_q.put(response)
            del mes

    def run(self, read_q, write_q):
        while not self.should_finish:
            self.handle_messages(read_q, write_q)
