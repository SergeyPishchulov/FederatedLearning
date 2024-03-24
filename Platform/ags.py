from torch.multiprocessing import Queue


class AGS:
    def __init__(self, user_args):
        self.user_args = user_args

    def run(self, hub_read_q, hub_write_q, q_by_cl_id):
