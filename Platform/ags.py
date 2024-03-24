from typing import Optional

from torch.multiprocessing import Queue
from datetime import datetime
import time

from message import MessageHubToAGS, ControlMessageHubToAGS


class AGS:
    def __init__(self, user_args):
        self.user_args = user_args
        self.start_time: Optional[datetime] = None
        self.should_finish = False

    def idle_until_run_cmd(self, hub_read_q):
        while self.start_time is None:
            self.handle_messages(hub_read_q)
            # print(f"Client {self.id} waiting run_cmd")
            time.sleep(1)

    def idle_until_start_time(self):
        if datetime.now() < self.start_time:
            delta = (self.start_time - datetime.now()).total_seconds()
            print(f"AGS WILL WAKE UP in {int(delta)}s")
            time.sleep(delta)

    def handle_messages(self, hub_read_q):
        while not hub_read_q.empty():  # 🍒
            mes = hub_read_q.get()
            if isinstance(mes, MessageHubToAGS):
                # TODO репозиторий тасок, куда будем добавлять новые. Основной цикл - решаем джобы они есть
                pass
            elif isinstance(mes, ControlMessageHubToAGS):
                self.start_time = mes.start_time
            else:
                raise ValueError(f"Unknown message {mes}")

    def run(self, hub_read_q, hub_write_q, q_by_cl_id):
        self.idle_until_run_cmd(hub_read_q)
        self.idle_until_run_cmd(hub_read_q)
        while not self.should_finish:
            self.handle_messages(hub_read_q)

