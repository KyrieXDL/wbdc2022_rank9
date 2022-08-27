# encoding=utf-8
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BertModel import BertModel
from category_id_map import CATEGORY_ID_LIST, LV1_CATEGORY_ID_LIST
from transformers import BertConfig
# visual backbones
from models.clip_vit import CLIPVisionModel
from transformers.models.clip.configuration_clip import CLIPVisionConfig

base_dir = os.path.dirname(__file__)
# torch.hub.set_dir(os.path.join(base_dir, '../../../data/cache'))

logger = logging.getLogger(__name__)


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.end2end = args.end2end
        self.backbone = args.backbone
        self.model_type = args.model_type
        self.frame_emb_type = args.frame_emb_type
        self.contras = args.contras
        self.fusion_layer = args.fusion_layer
        print('contras:', self.contras)
        
        self.visual_backbone = self.build_visual_encoder(args)
        # self.bert = BertModel.from_pretrained(args.bert_dir)
        # bert_output_size = 768
        bert_config = BertConfig.from_pretrained(os.path.join(args.bert_dir, 'config.json'))
        bert_weight = torch.load(os.path.join(args.bert_dir, 'pytorch_model.bin'))
        bert_state_dict = {}
        print(args.bert_dir)
        for k, v in bert_weight.items():
            bert_state_dict[k.replace('bert.', '')] = v
        self.bert = BertModel(config=bert_config)
        msg = self.bert.load_state_dict(bert_state_dict, strict=False)
        print(msg)

        bert_output_size = bert_config.hidden_size
        
        self.dropout = nn.Dropout(args.dropout)
        # 总体分类器
        self.classifier = nn.Linear(bert_output_size, len(CATEGORY_ID_LIST))

    def build_visual_encoder(self, args):
        if args.end2end:
            if args.backbone == 'clip_vit':
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
        else:
            visual_backbone = None
        return visual_backbone

    @staticmethod
    def get_mean_pool(sequence_output, attention_mask):
        return torch.einsum("bsh, bs, b->bh", sequence_output, attention_mask.float(),
                            1 / attention_mask.float().sum(dim=1) + 1e-9)

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
        # sequence_output1 = sequence_output
        # sequence_output = self.dropout(sequence_output)
        sequence_output = torch.mean(sequence_output, dim=1)

        sequence_output = self.dropout(sequence_output)

        prediction = self.classifier(sequence_output)

        if self.training:
            # loss, accuracy, pred_label_id, label = self.cal_loss(prediction, label)
            # embedding_text = sequence_output1[:, 0:256, :]
            # embedding_video = sequence_output1[:, 256:, :]
            itc_loss = None
            if self.contras:
                itc_loss = self.itc_loss(torch.mean(embedding_text, dim=1), torch.mean(embedding_video, dim=1))
            return prediction, itc_loss
        else:
            return prediction


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
