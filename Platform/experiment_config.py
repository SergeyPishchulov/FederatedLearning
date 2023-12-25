from argparse import Namespace


def get_configs(user_args):
    common_config = dict(interdeadline_time_sec=25,
                         required_quality=1_000)
    custom_configs = []
    custom_configs.append(dict(
        dataset='cifar10',
        local_model='CNN',
        target_acc=65,
    ))
    custom_configs.append(dict(
        dataset='cifar10',
        local_model='ResNet20',
        target_acc=65,
    ))
    custom_configs.append(dict(
        dataset='cifar10',
        local_model='ResNet56',
        target_acc=65,
    ))
    return [Namespace(**(vars(user_args) | common_config | cc)) for cc in custom_configs]


#
from itertools import zip_longest


def combine_lists(l):
    return [j for i in zip_longest(*l) for j in i if j]

# plan = combine_lists([
#     [(r, t) for r in range(10)] for t in ['t1', 't2', 't3']
# ])
# print(plan)
# def my_gen(n:int, cl):
#     for i in range(n):
#         yield (cl, f'round {i}')
#
# gens = [my_gen(10, 'cl1'), my_gen(8,'cl2')]
# for responses in zip(*gens):
#     for x in responses:
#         print(x)
#         print('**')
#     print('NEXT ROUND')

# ROUNDS = 2
# plan = combine_lists([
#     [(round, task) for round in range(ROUNDS)] for task in ['t1', 't2', 't3']
# ])
# print(f"Plan is {plan}")
