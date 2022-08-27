import torch
import torch.nn as nn
from models.backbone.defaults import get_cfg
from models.backbone.video_mvit import MViT
from models.backbone import video_swin
from models.backbone import swin
from models.backbone.clip_vit import CLIPVisionModel
from transformers.models.clip.configuration_clip import CLIPVisionConfig


class Model_Stack(nn.Module):
    def __init__(self, model_list, config, end2end=True):
        super(Model_Stack, self).__init__()
        self.frame_encoder_arch = config['frame_encoder_arch']
        self.frame_emb_type = config['frame_emb_type']
        self.model_list = nn.ModuleList(model_list)
        
        self.end2end = end2end
        if end2end:
            self.frame_encoder = self.build_frame_encoder(config)

    def build_frame_encoder(self, config):
        if self.frame_encoder_arch == 'mvit':
            cfg = get_cfg()
            cfg.merge_from_file(config['frame_encoder_config_path'])
            frame_encoder = MViT(cfg)
            state_dict = torch.load(config['frame_encoder_path'], map_location='cpu')
            msg = frame_encoder.load_state_dict(state_dict['model_state'], strict=False)
            print('mvit: ', msg)
        elif self.frame_encoder_arch == 'video_swin':
            frame_encoder = video_swin.video_swin('tiny', pretrained=config['frame_encoder_path'])
        elif self.frame_encoder_arch == 'swin':
            frame_encoder = swin.swin_tiny(config['frame_encoder_path'])
        elif self.frame_encoder_arch == 'clip_vit':
            clip_config = CLIPVisionConfig.from_pretrained(config['frame_encoder_config_path'])
            frame_encoder = CLIPVisionModel(clip_config)
            state_dict = torch.load(config['frame_encoder_path'], map_location='cpu')
            vit_state_dict = {}
            for k, v in state_dict.items():
                if 'text_model' in k:
                    continue
                vit_state_dict[k] = v
            msg = frame_encoder.load_state_dict(vit_state_dict, strict=False)
            print('clip vit: ', msg)

            if self.frame_emb_type == 'patch':
                frame_encoder.frozen_pooler_layer()
        else:
            raise NotImplementedError

        return frame_encoder

    def forward(self, frame_input, frame_mask, text_input_ids, text_attention_mask, text_segment_ids):
        visual_emb, visual_attn_mask = frame_input, frame_mask
        if self.end2end:
            visual_emb, visual_attn_mask = self.frame_encoder(frame_input, frame_mask)
        output_list = []
        for model in self.model_list:
            output = model(visual_emb, visual_attn_mask, text_input_ids, text_attention_mask, text_segment_ids)
            output_list.append(output)
        
        return output_list



