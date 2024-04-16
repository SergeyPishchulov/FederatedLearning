import copy
from typing import List, Optional, Dict

from utils import get_params_cnt
from models_dict.reparam_function import ReparamModule
import torch.nn as nn


class ModelType:
    FEDLAW = 'fedlaw'
    ORDIANRY = 'ordinary'


class ModelTypedState:
    def __init__(self, model_type, state, params_cnt):
        self.model_type = model_type
        self.state = state
        self.params_cnt = params_cnt


def typed_states_to_states(states: List[ModelTypedState]):
    return [x.state for x in states]


class ModelCast:
    @staticmethod
    def to_state(model: nn.Module) -> ModelTypedState:
        if isinstance(model, ReparamModule):
            params_cnt = get_params_cnt(model)
            return ModelTypedState(ModelType.FEDLAW,
                                   state=copy.deepcopy(
                                       model.get_param(clone=True).cpu().detach()),  # TODO .cpu() ?
                                   params_cnt=params_cnt)
        elif isinstance(model, nn.Module):
            params_cnt = get_params_cnt(model)
            return ModelTypedState(ModelType.ORDIANRY, copy.deepcopy(model.state_dict()),
                                   params_cnt=params_cnt)
        else:
            raise ValueError(f"Unknown model type {type(model)}")

    @staticmethod
    def to_model(typed_state: ModelTypedState, model_to_write: nn.Module):
        if not isinstance(typed_state, ModelTypedState):
            raise TypeError('State must be a ModelState')
        if typed_state.model_type == ModelType.FEDLAW:
            return model_to_write.load_param(copy.deepcopy(typed_state.state))
        if typed_state.model_type == ModelType.ORDIANRY:
            return model_to_write.load_state_dict(typed_state.state)
        raise ValueError(f"Unknown typed_state {typed_state}")
