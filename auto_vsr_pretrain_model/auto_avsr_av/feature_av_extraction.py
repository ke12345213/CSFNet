import sys
sys.path.insert(0, "../")

import os
import torch

import argparse
parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args(args=[])

from espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E
from pytorch_lightning import LightningModule
from datamodule.transforms import TextTransform

from argparse import Namespace

config_dict = {
    "adim": 768,
    "aheads": 12,
    "eunits": 3072,
    "elayers": 12,
    "transformer_input_layer": "conv3d",
    "dropout_rate": 0.1,
    "transformer_attn_dropout_rate": 0.1,
    "transformer_encoder_attn_layer_type": "rel_mha",
    "macaron_style": True,
    "use_cnn_module": True,
    "cnn_module_kernel": 31,
    "zero_triu": False,
    "a_upsample_ratio": 1,
    "relu_type": "swish",
    "ddim": 768,
    "dheads": 12,
    "dunits": 3072,
    "dlayers": 6,
    "lsm_weight": 0.1,
    "transformer_length_normalized_loss": False,
    "mtlalpha": 0.1,
    "ctc_type": "builtin",
    "rel_pos_type": "latest",

    "aux_adim": 768,
    "aux_aheads": 12,
    "aux_eunits": 3072,
    "aux_elayers": 12,
    "aux_transformer_input_layer": "conv1d",
    "aux_dropout_rate": 0.1,
    "aux_transformer_attn_dropout_rate": 0.1,
    "aux_transformer_encoder_attn_layer_type": "rel_mha",
    "aux_macaron_style": True,
    "aux_use_cnn_module": True,
    "aux_cnn_module_kernel": 31,
    "aux_zero_triu": False,
    "aux_a_upsample_ratio": 1,
    "aux_relu_type": "swish",
    "aux_dunits": 3072,
    "aux_dlayers": 6,
    "aux_lsm_weight": 0.1,
    "aux_transformer_length_normalized_loss": False,
    "aux_mtlalpha": 0.1,
    "aux_ctc_type": "builtin",
    "aux_rel_pos_type": "latest",

    "fusion_hdim": 8192,
    "fusion_norm": "batchnorm",
}

args = Namespace(**config_dict)


class ModelModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # self.modality = args.modality
        self.adim = args.adim
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        self.model = E2E(len(self.token_list), args, ignore_id=-1)

    def forward(self, x, a):
        x, _ = self.model.encoder(x, None)
        # print(x.shape)  # torch.Size([4, 50, 768]) 4是batch
        a, _ = self.model.aux_encoder(a, None)  
        # print(a.shape)  # # torch.Size([4, 50, 768])
        f = self.model.fusion(torch.cat((x, a), dim=-1))
        # print(f.shape)  # torch.Size([4, 50, 768])
        return x, a, f
    
model_path = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/avsr_trlrwlrs2lrs3vox2avsp_base.pth"

model = ModelModule(args)
ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
model.model.load_state_dict(ckpt)
model.freeze()

x = torch.randn((4, 50, 1, 88, 88))
a = torch.randn((4, 32000, 1)) 
with torch.inference_mode():
    video, audio, y = model(x, a)
print(y.size())