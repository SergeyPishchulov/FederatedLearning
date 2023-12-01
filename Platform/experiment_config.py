from argparse import Namespace

def get_configs(user_args):
    custom_configs = []
    custom_configs.append(dict(
        dataset='cifar10',
        local_model='CNN',
        epochs='2')
    )
    custom_configs.append(dict(
        dataset='cifar10',
        local_model='CNN',
        epochs='1')
    )
    return [Namespace(**(vars(user_args) | cc)) for cc in custom_configs]
