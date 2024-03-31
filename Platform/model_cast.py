import copy

from reparam_function import ReparamModule
import torch.nn as nn


class ModelType:
    FEDLAW = 'fedlaw'
    ORDIANRY = 'ordinary'


class ModelTypedState:
    def __init__(self, model_type, state):
        self.model_type = model_type
        self.state = state


class ModelCast:
    @staticmethod
    def to_state(model):
        if isinstance(model, ReparamModule):
            return ModelTypedState(ModelType.FEDLAW,
                                   state=model.get_param(clone=True))
        elif isinstance(model, nn.Module):
            return ModelTypedState(ModelType.ORDIANRY, copy.deepcopy(model.state_dict()))

    @staticmethod
    def to_model(typed_state: ModelTypedState, model_to_write: nn.Module):
        if not isinstance(typed_state, ModelTypedState):
            raise TypeError('State must be a ModelState')
        if typed_state.model_type == ModelType.FEDLAW:
            # TODO in case of memory sharing consider copy.deepcopy of model
            return model_to_write.load_param(copy.deepcopy(typed_state.state))
        if typed_state.model_type == ModelType.ORDIANRY:
            return model_to_write.load_param(typed_state.state)
        raise ValueError(f"Unknown typed_state {typed_state}")
