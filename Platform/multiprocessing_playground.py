# # fed_tasks = [...]
# #
# # clients = [Client(get_nodes_by_ft_task(ft))
# #            for ft in fed_tasks]
# #
# # for c in clients:
# #     c.run()
# #
# # train_journal = []
# # while not all_done:
# #     for mes in client_pipes:
# #         train_journal.append(parse_update(mes))
# #
# #     for tf in fed_tasks:
# #         if ready_for_aggreagation(tf, train_journal)
# #             ags_pipe.request_agg_model(train_journal(ft).get_parameters())
# #
# #     for mes in ags_pipe:
# #         ft, agr_model = parse(mes)
# #         for c in clients:
# #             c.pipe.send_agr_model(ft, agr_model)
# #
# #     sleep(5)
# #
# #
# import multiprocessing
# import time
#
#
# def worker(read, write):
#     state = 0
#     name_proc = multiprocessing.current_process().name
#     for x in range(100):
#         if not read.empty():
#             new = read.get()
#             state = new
#         write.put(f"{state}_{name_proc}")
#         time.sleep(1)
#     read.close()
#     write.close()
#
#
# write = multiprocessing.Queue()
# read = multiprocessing.Queue()
# # [read.put(x) for x in range(3, 7)]
#
# NUM_CORE = 2
# procs = []
# for i in range(NUM_CORE):
#     p = multiprocessing.Process(target=worker, args=(read, write,))
#     procs.append(p)
#     p.start()
#
# for y in range(100):
#     if not write.empty():
#         print(write.get())
#     if y % 5 == 0:
#         read.put(y)
#     time.sleep(1)
#
# [proc.join() for proc in procs]
# print([write.get() for _ in range(write.qsize())])
from Platform.utils import divide_almost_equally

print(divide_almost_equally(199, 5))