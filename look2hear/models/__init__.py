from .IIANet import IIANet
from .restormer_1d import restormer_1d
from .restormer_2d import restormer_2d
from .restormer_2d_v2 import restormer_2d_v2
from .DPT_1d import DPT_1d
from .crossnet import crossnet
from .tfgridnet import tfgridnet
from .tfgridnet_v2 import tfgridnet_v2
from .tfgridnet_v3 import tfgridnet_v3
from .tfgridnet_v2_step2 import tfgridnet_v2_step2
from .tfgridnet_v2_a_o import tfgridnet_v2_a_o
from .tfgridnet_v2_wer import tfgridnet_v2_wer
from .tfgridnet_v2_Lip_o import tfgridnet_v2_Lip_o
from .tfgridnet_v2_face_o import tfgridnet_v2_face_o

__all__ = [
    "IIANet",
    "restormer_1d"
    "restormer_2d"
    "restormer_2d_v2"
    "DPT_1d"
    "crossnet"
    "tfgridnet_v3"
    "tfgridnet_v2"
    "tfgridnet_v2_step2"
    "tfgridnet_v2_a_o"
    "tfgridnet_v2_wer"
    "tfgridnet_v2_Lip_o"
    "tfgridnet_v2_face_o"
]


def register_model(custom_model):
    """Register a custom model, gettable with `models.get`.

    Args:
        custom_model: Custom model to register.

    """
    if (
        custom_model.__name__ in globals().keys()
        or custom_model.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Model {custom_model.__name__} already exists. Choose another name."
        )
    globals().update({custom_model.__name__: custom_model})


def get(identifier):
    """Returns an model class from a string (case-insensitive).

    Args:
        identifier (str): the model name.

    Returns:
        :class:`torch.nn.Module`
    """
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")
