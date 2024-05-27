from argparse import Namespace


def get_configs(user_args):
    common_config = dict(
        required_quality=2_000,
        interdeadline_time_sec=25

    )
    custom_configs = []
    custom_configs.append(dict(
        dataset='cifar10',
        local_model='CNN',
        target_acc=50,
        # interdeadline_time_sec=25
    ))
    custom_configs.append(dict(
        dataset='cifar100',
        local_model='ResNet20',
        target_acc=20,
        # interdeadline_time_sec=30
    ))
    custom_configs.append(dict(
        dataset='cifar10',
        local_model='ResNet56',
        target_acc=50,
        # interdeadline_time_sec=30
    ))
    custom_configs.append(dict(
        dataset='cifar10',
        local_model='ResNet56',#'WRN56_2',
        target_acc=50,
        # interdeadline_time_sec=30
    ))
    custom_configs.append(dict(
        dataset='cifar100',
        local_model='WRN56_2',
        target_acc=20,
        # interdeadline_time_sec=30
    ))
    #######################################################################
    custom_configs.append(dict(
        dataset='cifar100',
        local_model='CNN',
        target_acc=50,
        # interdeadline_time_sec=25
    ))
    # custom_configs.append(dict(
    #     dataset='cifar100',
    #     local_model='ResNet20',
    #     target_acc=65,
    # ))
    #
    # custom_configs.append(dict(
    #     dataset='cifar100',
    #     local_model='ResNet56',
    #     target_acc=65,
    # ))
    # custom_configs.append(dict(
    #     dataset='cifar100',
    #     local_model='ResNet56',#'WRN56_2',
    #     target_acc=65,
    # ))
    # custom_configs.append(dict(
    #     dataset='cifar100',
    #     local_model='WRN56_2',
    #     target_acc=65,
    # ))

    return [Namespace(**(vars(user_args) | common_config | cc))
            for cc in custom_configs]


#
from itertools import zip_longest


def combine_lists(l):
    return [j for i in zip_longest(*l) for j in i if j]
