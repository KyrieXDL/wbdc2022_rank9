# encoding=utf-8
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import BertModel, VisualBertModel, VisualBertConfig
from models.BertModel import BertModel
from models.swin import swin_tiny
from category_id_map import CATEGORY_ID_LIST, LV1_CATEGORY_ID_LIST

# visual backbones
from models.efficientNet import EfficientNet
from models.van import *
from models.convnext import *
from models.clip_vit import CLIPVisionModel
from transformers.models.clip.configuration_clip import CLIPVisionConfig

base_dir = os.path.dirname(__file__)
torch.hub.set_dir(os.path.join(base_dir, '../../../data/cache'))

logger = logging.getLogger(__name__)


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.end2end = args.end2end
        self.backbone = args.backbone
        self.model_type = args.model_type
        self.frame_emb_type = args.frame_emb_type
        
        self.visual_backbone = self.build_visual_encoder(args)
        self.bert = BertModel.from_pretrained(args.bert_dir)

        bert_output_size = 768
        self.dropout = nn.Dropout(args.dropout)
        # self.dropout = SpatialDropout(args.dropout)
        # self.vision_proj = nn.Linear(bert_output_size, bert_output_size)
        # self.text_proj = nn.Linear(bert_output_size, bert_output_size)
            
        # self.text_linear = nn.Linear(bert_output_size, bert_output_size)
        # self.image_linear = nn.Linear(bert_output_size, bert_output_size)

        # 总体分类器
        self.classifier = nn.Linear(bert_output_size, len(CATEGORY_ID_LIST))

    def build_visual_encoder(self, args):
        if args.end2end:
            if args.backbone == 'swin-tiny':
                visual_backbone = swin_tiny(args.swin_pretrained_path)  # swin-tiny
            elif args.backbone == 'efficientnet':
                visual_backbone = EfficientNet.from_pretrained('efficientnet-b0', num_classes=768,
                                                               image_size=args.input_shape,
                                                               dropout_rate=args.dropout)  # efficientNet
            elif args.backbone == 'van':
                visual_backbone = van_small(pretrained=True, num_classes=768)  # van
            elif args.backbone == 'convnext':
                visual_backbone = convnext_tiny(pretrained=True, num_classes=768)  # convnext
            elif args.backbone == 'clip_vit':
                clip_config = CLIPVisionConfig.from_pretrained(args.frame_encoder_config_path)
                visual_backbone = CLIPVisionModel(clip_config)
                state_dict = torch.load(args.frame_encoder_path, map_location='cpu')
                vit_state_dict = {}
                for k, v in state_dict.items():
                    if 'text_model' in k:
                        continue
                    vit_state_dict[k] = v
                msg = visual_backbone.load_state_dict(vit_state_dict, strict=False)
                print('clip vit: ', msg)

                if self.frame_emb_type == 'patch':
                    visual_backbone.frozen_pooler_layer()
        return visual_backbone

    def itc_loss(self, embedding_text, sembedding_video):

        # embedding_text = F.normalize(self.text_proj(embedding_text), dim=-1)
        # embedding_video = F.normalize(self.vision_proj(sembedding_video), dim=-1)
        embedding_text = F.normalize(embedding_text, dim=-1)
        embedding_video = F.normalize(sembedding_video, dim=-1)

        # 图像到文本
        sim_i2t = 100.0 * embedding_video @ embedding_text.T
        sim_t2i = 100.0 * embedding_text @ embedding_video.T

        sim_targets = torch.zeros(sim_i2t.size()).to(embedding_video.device)
        sim_targets.fill_diagonal_(1)

        alpha = 0
        sim_i2t_targets = alpha * F.softmax(sim_i2t, dim=1) + (1 - alpha) * sim_targets
        sim_t2i_targets = alpha * F.softmax(sim_t2i, dim=1) + (1 - alpha) * sim_targets

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        return (loss_i2t + loss_t2i) / 2

    def forward(self, frame_input, frame_mask, title_input, title_mask, token_type_ids):
        
        if self.end2end:
            if self.backbone == 'clip_vit':
                frame_input, frame_mask = self.visual_backbone(frame_input, frame_mask, emb_type=self.frame_emb_type)
            else:
                frame_input = self.visual_backbone(frame_input)

        # [batch_size, seq_len, dim]
        vision_feature, vision_mask = frame_input, frame_mask

        sequence_output, pooled_output, hidden_state, embedding_text, embedding_video = self.bert(
            input_ids=title_input, text_mask=title_mask, token_type_ids=token_type_ids,
            video_mask=vision_mask, video_embedding=vision_feature)

        # sequence_output = self.dropout(sequence_output)
        sequence_output = torch.mean(sequence_output, dim=1)

        sequence_output = self.dropout(sequence_output)

        prediction = self.classifier(sequence_output)
        
        return prediction


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, out_size=768, linear_layer_size=[1024, 512], num_label=200, hidden_dropout_prob=0.2):
        super().__init__()
        self.norm = nn.BatchNorm1d(out_size)
        self.dense = nn.Linear(out_size, linear_layer_size[0])
        self.norm_1 = nn.BatchNorm1d(linear_layer_size[0])
        # self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dense_1 = nn.Linear(linear_layer_size[0], linear_layer_size[1])
        self.norm_2 = nn.BatchNorm1d(linear_layer_size[1])
        self.out_proj = nn.Linear(linear_layer_size[1], num_label)

    def forward(self, features, **kwargs):
        x = self.norm(features)
        # x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(self.norm_1(x))
        # x = self.dropout(x)
        x = self.dense_1(x)
        x = torch.relu(self.norm_2(x))
        # x = self.dropout(x)
        x = self.out_proj(x)
        return x


import math
from itertools import repeat


class SpatialDropout(nn.Module):
    """
    对字级别的向量进行丢弃
    """

    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(input.size(0), *repeat(1, input.dim() - 2), input.size(2))

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output


if __name__ == '__main__':
    from config import parse_args

    args = parse_args()

    title_input = torch.randint(1, 1000, (2, 128))  # (batch x frames x channels x height x width)
    frame_input = torch.randn(2, 32, 768)  # (batch x frames x channels x height x width)
    title_mask = torch.ones(2, 128)
    frame_mask = torch.ones(2, 32)
    data = dict(
        frame_input=frame_input,
        frame_mask=frame_mask,
        title_input=title_input,
        title_mask=title_mask,

    )

    model = MultiModal(args)
    model(data)
