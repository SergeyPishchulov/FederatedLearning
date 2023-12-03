from argparse import Namespace


def get_configs(user_args):
    custom_configs = []
    custom_configs.append(dict(
        dataset='cifar10',
        local_model='CNN',
    )
    )
    custom_configs.append(dict(
        dataset='cifar10',
        local_model='ResNet20',
    )
    )
    return [Namespace(**(vars(user_args) | cc)) for cc in custom_configs]

#
# from itertools import zip_longest
#
#
# def combine_lists(l):
#     return [j for i in zip_longest(*l) for j in i if j]
#
#
# plan = combine_lists([
#     [(r, t) for r in range(10)] for t in ['t1', 't2', 't3']
# ])
# print(plan)
