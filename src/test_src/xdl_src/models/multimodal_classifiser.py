import torch
import torch.nn as nn
from transformers.models.bert import BertModel, BertTokenizer, BertConfig
from models.multimodal_encoder import MultiModalEncoder
import os
from transformers.models.vit import ViTModel, ViTConfig
# from models.visual_encoder import VisualEncoder
from models.visual_encoder import VisualEncoder_Postnorm, VisualEncoder_Prenorm
from models.model_modules import ConcatDenseSE, NeXtVLAD, SpatialDropout
import torch.nn.functional as F
import torch.distributed as dist
import sys
from models.backbone.defaults import get_cfg
from models.backbone.video_mvit import MViT
from models.backbone import video_swin
from models.backbone import swin
from models.backbone.clip_vit import CLIPVisionModel
from transformers.models.clip.configuration_clip import CLIPVisionConfig


class MultiModal_Classifier(nn.Module):
    def __init__(self, config):
        super(MultiModal_Classifier, self).__init__()
        self.config = config
        self.use_visual_encoder = config['use_visual_encoder']
        self.fusion = config['fusion']
        self.use_asr = config['use_asr']
        self.use_ocr = config['use_ocr']
        self.use_tfidf = config['use_tfidf']
        self.max_title_len = config['max_title_len']
        self.max_asr_len = config['max_asr_len']
        self.max_ocr_len = config['max_ocr_len']
        self.asr_type = config['asr_type']
        self.ocr_type = config['ocr_type']
        self.max_len = min(self.max_title_len + 2, config['max_len'])
        self.use_raw_image = config['use_raw_image']
        if self.use_asr:
            if self.use_ocr:
                self.max_len = min(self.max_title_len + self.max_asr_len + self.max_ocr_len + 4, config['max_len'])
            else:
                self.max_len = min(self.max_title_len + self.max_asr_len + 3, config['max_len'])
        self.cross_type = config['cross_type']
        self.frame_encoder_arch = config['frame_encoder_arch']
        self.visual_encoder_arch = config['visual_encoder_arch']
        self.use_prompt = config['use_prompt']
        self.use_pooling = config['use_pooling']
        self.pooling = config['pooling']
        self.use_lv1 = config['use_lv1']
        self.use_contrastive = config['use_contrastive']
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.truncation = config['truncation']
        self.use_single_modal = config['use_single_modal']
        self.frame_emb_type = config['frame_emb_type']

        if self.use_raw_image:
            self.frame_encoder = self.build_frame_encoder(config)

        if self.use_visual_encoder:
            self.visual_encoder = self.build_visual_encoder(config)
            
        if self.use_contrastive:
            self.vision_proj = nn.Linear(self.config['mm_embed_dim'], self.config['mm_embed_dim'])
            self.text_proj = nn.Linear(self.config['mm_embed_dim'], self.config['mm_embed_dim'])
            # self.temp = nn.Parameter(torch.ones([]) * 0.07) 

        self.tokenizer, self.text_encoder = self.build_text_encoder(config)

        if self.fusion in ['merge_attention', 'bottleneck_attention']:
            self.multimodal_encoder = self.build_multi_modal_encoder(config)
        else:
            if 'image' in self.cross_type and 'text' in self.cross_type:
                self.multimodal_encoder_image = self.build_multi_modal_encoder(config)
                self.multimodal_encoder_text = self.build_multi_modal_encoder(config)
            elif 'image' in self.cross_type:
                self.multimodal_encoder_image = self.build_multi_modal_encoder(config)
            else:
                self.multimodal_encoder_text = self.build_multi_modal_encoder(config)

        if self.pooling == 'weight':
            self.fusion_layer = nn.Linear(self.config['mm_embed_dim'], 1)
        elif self.pooling == 'enhance':
            self.fusion_layer = ConcatDenseSE(self.config['mm_embed_dim'] * 2, self.config['mm_embed_dim'])
        self.spatial_dropout = SpatialDropout(config['spatial_dropout'])
        #         self.spatial_dropout = nn.Dropout(p=config['spatial_dropout'])

        if self.use_lv1:
            self.lv1_classifier_head = nn.Linear(self.config['mm_embed_dim'], self.config['label1_nums'])

        ratio = 1
        if self.cross_type == 'image_text':
            ratio = 2
        single_modal_dim = 0
        if self.use_single_modal:
            single_modal_dim = 768 * 2
        self.lv2_classifier_head = nn.Linear(self.config['mm_embed_dim'] * ratio + single_modal_dim,
                                             self.config['label2_nums'])

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

    def build_visual_encoder(self, config):
        if self.visual_encoder_arch == 'transformer_prenorm':
            visual_config = ViTConfig.from_pretrained(config['visual_encoder_config_path'])
            visual_encoder = VisualEncoder_Prenorm(visual_config)
        elif self.visual_encoder_arch == 'transformer_postnorm':
            visual_config = BertConfig.from_pretrained(config['visual_encoder_config_path'])
            visual_encoder = VisualEncoder_Postnorm(visual_config)
        elif self.visual_encoder_arch == 'nextvlad':
            visual_encoder = NeXtVLAD(768, 64, output_size=config['visual_embed_dim'], dropout=0.3)
        else:
            raise NotImplementedError

        return visual_encoder

    def build_text_encoder(self, config):
        bert_config = BertConfig.from_pretrained(os.path.join(config['text_encoder_path'], 'config.json'))
        # bert_weight = torch.load(os.path.join(config['text_encoder_path'], 'pytorch_model.bin'))
        # bert_state_dict = {}
        # for k, v in bert_weight.items():
        #     bert_state_dict[k.replace('bert.', '')] = v
        if config['text_encoder_arch'] == 'bert':
            tokenizer = BertTokenizer.from_pretrained(config['text_encoder_path'])
            text_encoder = BertModel(config=bert_config, add_pooling_layer=False)
            # msg = text_encoder.load_state_dict(bert_state_dict, strict=False)
        else:
            raise NotImplementedError
        # print('load bert weight: ', msg)
        return tokenizer, text_encoder

    def build_multi_modal_encoder(self, config):
        if 'attention' in self.fusion:
            multimodal_config = BertConfig.from_pretrained(config['multimodal_config_path'])
            multimodal_encoder = MultiModalEncoder(multimodal_config)
        else:
            multimodal_encoder = ConcatDenseSE(config['visual_embed_dim'] + config['text_embed_dim'],
                                               config['mm_embed_dim'])

        return multimodal_encoder

    def frozen_bert_layers(self, layer=9):
        frozen_params = ['text_encoder.encoder.layer.{}.'.format(i) for i in range(layer)]
        for n, p in self.named_parameters():
            if any(fn in n for fn in frozen_params):
                p.requires_grad = False

    def forward(self, frame_feats, frame_feats_mask, text_input_ids, text_attention_mask, text_segment_ids):
        # print(frame_feats.shape, frame_feats_mask.shape, len(title_text))
        ### 文本处理
        # input_ids, segment_ids, attention_mask, pool_mask = self.process_text(title_text, asr_text, ocr_text)
        # input_ids, segment_ids, attention_mask = input_ids.to(frame_feats.device), segment_ids.to(
        #     frame_feats.device), attention_mask.to(frame_feats.device)
        # print('device: ', text_input_ids.device, text_segment_ids.device, text_attention_mask.device)
        # frame_feats_mask = frame_feats_mask.float()
        # text_attention_mask = text_attention_mask.float()
        

        ### 提取文本特征
        text_emb = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask, token_type_ids=text_segment_ids)[0]
        text_attn_mask = text_attention_mask

        ### 提取帧特征
        visual_attn_mask = frame_feats_mask
        visual_emb = frame_feats
        if self.use_raw_image:
            visual_emb, visual_attn_mask = self.frame_encoder(visual_emb, visual_attn_mask, emb_type=self.frame_emb_type)
            
        ### 不同帧进行交互
        if self.use_visual_encoder:
            visual_emb, visual_attn_mask = self.visual_encoder(visual_emb, visual_attn_mask)

        
        ### 对比学习
        loss_itc = None
        if self.use_contrastive and self.training:
            cls_visual_emb = F.normalize(self.vision_proj(visual_emb[:, 0, :]), dim=-1)
            cls_text_emb = F.normalize(self.text_proj(text_emb[:, 0, :]), dim=-1)
            # 计算图像文本之间的相似度
            sim_i2t = cls_visual_emb @ cls_text_emb.T #/ self.temp
            sim_t2i = cls_text_emb @ cls_visual_emb.T #/ self.temp
            sim_targets = torch.zeros(sim_i2t.size()).to(visual_emb.device)
            sim_targets.fill_diagonal_(1)
            # image text contrastive 的交叉熵loss
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()
            loss_itc = (loss_i2t + loss_t2i) / 2

        ### 不同模态交互
        if self.fusion == 'merge_attention':
            multimodal_emb = self.multimodal_encoder(
                hidden_states=torch.cat([visual_emb, text_emb], dim=1),
                attention_mask=torch.cat([visual_attn_mask, text_attn_mask], dim=1))[0]
            visual_cls_emb = multimodal_emb[:, 0, :]
            text_cls_emb = multimodal_emb[:, visual_emb.size()[1], :]
            if self.pooling != '':
                multimodal_emb = self.emb_pool([visual_cls_emb, text_cls_emb], pooling=self.pooling)
            else:
                multimodal_emb = text_cls_emb

        elif self.fusion == 'bottleneck_attention':
            output = self.multimodal_encoder(
                hidden_states=visual_emb,
                attention_mask=visual_attn_mask,
                encoder_hidden_states=text_emb,
                encoder_attention_mask=text_attn_mask,
                output_hidden_states=True)
            visual_cls_emb = output.last_hidden_state[:, 0, :]
            text_cls_emb = output.last_hidden_state[:, -text_emb.size()[1], :]
            if self.pooling != '':
                multimodal_emb = self.emb_pool([visual_cls_emb, text_cls_emb], pooling=self.pooling)
            else:
                multimodal_emb = text_cls_emb

        elif self.fusion == 'cross_attention':
            if self.cross_type == 'text':
                output = self.multimodal_encoder_text(
                    hidden_states=text_emb,
                    attention_mask=text_attn_mask,
                    encoder_hidden_states=visual_emb,
                    encoder_attention_mask=visual_attn_mask,
                    output_hidden_states=True)

                if self.pooling != '':
                    # emb_list = [output.hidden_states[-i][:, 0, :] for i in range(1, 5)]
                    emb_list = output.last_hidden_state
                    multimodal_emb = self.emb_pool(emb_list, pooling=self.pooling)
                else:
                    multimodal_emb = output.last_hidden_state[:, 0, :]

            elif self.cross_type == 'image':
                output = self.multimodal_encoder_image(
                    hidden_states=visual_emb,
                    attention_mask=visual_attn_mask,
                    encoder_hidden_states=text_emb,
                    encoder_attention_mask=text_attn_mask,
                    output_hidden_states=True)

                if self.pooling != '':
                    emb_list = [output.hidden_states[-i][:, 0, :] for i in range(1, 5)]
                    multimodal_emb = self.emb_pool(emb_list, pooling=self.pooling)
                else:
                    multimodal_emb = output.last_hidden_state[:, 0, :]
            else:
                # visual_emb = visual_emb.float()
                # visual_attn_mask = visual_attn_mask.float()
                output_text = self.multimodal_encoder_text(
                    hidden_states=text_emb,
                    attention_mask=text_attn_mask,
                    encoder_hidden_states=visual_emb,
                    encoder_attention_mask=visual_attn_mask,
                    output_hidden_states=True)
                output_image = self.multimodal_encoder_image(
                    hidden_states=visual_emb,
                    attention_mask=visual_attn_mask,
                    encoder_hidden_states=text_emb,
                    encoder_attention_mask=text_attn_mask,
                    output_hidden_states=True)

                if self.pooling != '':
                    # emb_text_list = [output_text.hidden_states[-i][:, 0, :] for i in range(1, 3)]
                    # emb_image_list = [output_image.hidden_states[-i][:, 0, :] for i in range(1, 3)]

                    emb_text_list = output_text.last_hidden_state
                    emb_image_list = output_image.last_hidden_state
                    multimodal_emb_text = self.emb_pool(emb_text_list, pooling=self.pooling)
                    multimodal_emb_image = self.emb_pool(emb_image_list, pooling=self.pooling)

                    multimodal_emb = torch.cat([multimodal_emb_image, multimodal_emb_text], dim=-1)
                #                     multimodal_emb = self.emb_pool([multimodal_emb_image, multimodal_emb_text], pooling=self.pooling)
                else:
                    multimodal_emb_text = output_text.last_hidden_state[:, 0, :]
                    multimodal_emb_image = output_image.last_hidden_state[:, 0, :]
                    multimodal_emb = torch.cat([multimodal_emb_image, multimodal_emb_text], dim=-1)
        else:
            raise NotImplementedError

        # if self.use_tfidf and tfidf is not None:
        #     multimodal_emb = self.emb_pool([multimodal_emb, tfidf], pooling=self.pooling)

        if self.use_single_modal:
            if self.pooling != '':
                visual_emb = self.emb_pool(visual_emb, pooling='mean')
                text_emb = self.emb_pool(text_emb, pooling='mean')
            else:
                visual_emb = visual_emb[:, 0, :]
                text_emb = text_emb[:, 0, :]
            multimodal_emb = torch.cat([multimodal_emb, visual_emb, text_emb], dim=-1)

        output1 = None
        if self.use_lv1:
            output1 = self.lv1_classifier_head(multimodal_emb)
        output2 = self.lv2_classifier_head(multimodal_emb)
        
        if not self.training:
            return output2
        return (output2, output1, loss_itc)

    def emb_pool(self, emb_list, pooling=''):
        if type(emb_list) == list:
            emb = torch.stack(emb_list, dim=1)
        else:
            emb = emb_list

        if pooling == 'max':
            pool_emb = torch.max(self.spatial_dropout(emb), dim=1)[0]
        elif pooling == 'mean':
            pool_emb = torch.mean(self.spatial_dropout(emb), dim=1)
        elif pooling == 'weight':
            emb_w = torch.softmax(self.fusion_layer(emb), dim=1)
            pool_emb = torch.sum(emb * emb_w, dim=1)
        elif pooling == 'enhance':
            pool_emb = self.fusion_layer(emb_list)
        else:
            raise ValueError
        return pool_emb