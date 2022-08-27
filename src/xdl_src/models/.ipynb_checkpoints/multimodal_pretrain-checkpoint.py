import torch
import torch.nn as nn
from transformers.models.bert import BertModel, BertTokenizer, BertConfig
from models.multimodal_encoder import MultiModalEncoder
import os
from transformers.models.vit import ViTModel, ViTConfig
from models.visual_encoder import VisualEncoder_Prenorm, VisualEncoder_Postnorm
from models.model_modules import ConcatDenseSE, NeXtVLAD, MLMHead, MFMHead
import torch.nn.functional as F
import torch.distributed as dist
from models.backbone.defaults import get_cfg
from models.backbone.video_mvit import MViT
from models.backbone import video_swin
from models.backbone import swin
from models.backbone.clip_vit import CLIPVisionModel
from transformers.models.clip.configuration_clip import CLIPVisionConfig


class MultiModal_Pretrain(nn.Module):
    def __init__(self, config):
        super(MultiModal_Pretrain, self).__init__()
        self.config = config
        self.use_visual_encoder = config['use_visual_encoder']
        self.fusion = config['fusion']
        self.cross_type = config['cross_type']
        self.visual_encoder_arch = config['visual_encoder_arch']
        self.use_prompt = config['use_prompt']
        self.mlm_probability = config['mlm_probability']
        self.mfm_probability = config['mfm_probability']
        self.mm_embed_dim = config['mm_embed_dim']
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.vocab_size = 0
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.pooling = config['pooling']
        self.visual_cls = False
        self.use_raw_image = config['use_raw_image']
        self.frame_encoder_arch = config['frame_encoder_arch']
        self.frame_emb_type = config['frame_emb_type']
        self.use_hard_negs = config['use_hard_negs']

        if self.use_raw_image:
            self.frame_encoder = self.build_frame_encoder(config)

        if self.use_visual_encoder:
            self.visual_encoder = self.build_visual_encoder(config)
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

        if 'itc' in config['tasks']:
            self.vision_proj = nn.Linear(self.mm_embed_dim, self.mm_embed_dim)
            self.text_proj = nn.Linear(self.mm_embed_dim, self.mm_embed_dim)

        if 'itm' in config['tasks']:
            self.itm_head = nn.Linear(self.mm_embed_dim * 2, 2)

        if 'mlm' in config['tasks']:
            self.mlm_head = MLMHead(config=BertConfig(vocab_size=self.vocab_size))

        if 'mfm' in config['tasks']:
            self.mfm_head = MFMHead()

        # momentum modules
        if self.use_visual_encoder:
            self.visual_encoder_m = self.build_visual_encoder(config)
        _, self.text_encoder_m = self.build_text_encoder(config)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.text_encoder, self.text_encoder_m],
        ]

        if self.use_raw_image:
            self.frame_encoder_m = self.build_frame_encoder(config)
            self.model_pairs += [[self.frame_encoder, self.frame_encoder_m]]

        if 'itc' in config['tasks']:
            self.vision_proj_m = nn.Linear(self.mm_embed_dim, self.mm_embed_dim)
            self.text_proj_m = nn.Linear(self.mm_embed_dim, self.mm_embed_dim)
            self.model_pairs += [[self.vision_proj, self.vision_proj_m],
                                 [self.text_proj, self.text_proj_m]]

        self.copy_params()

        # create the queue
        self.register_buffer("vision_queue", torch.randn(self.mm_embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.mm_embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.vision_queue = nn.functional.normalize(self.vision_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

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
            self.visual_cls = visual_config.add_cls
        elif self.visual_encoder_arch == 'transformer_postnorm':
            visual_config = BertConfig.from_pretrained(config['visual_encoder_config_path'])
            visual_encoder = VisualEncoder_Postnorm(visual_config)
            self.visual_cls = visual_config.add_cls
        elif self.visual_encoder_arch == 'nextvlad':
            visual_encoder = NeXtVLAD(768, 64, output_size=config['visual_embed_dim'], dropout=0.3)
        else:
            raise NotImplementedError

        return visual_encoder

    def build_text_encoder(self, config):
        bert_config = BertConfig.from_pretrained(os.path.join(config['text_encoder_path'], 'config.json'))
        bert_weight = torch.load(os.path.join(config['text_encoder_path'], 'pytorch_model.bin'))
        bert_state_dict = {}
        self.vocab_size = bert_config.vocab_size
        for k, v in bert_weight.items():
            bert_state_dict[k.replace('bert.', '')] = v
        if config['text_encoder_arch'] == 'bert':
            tokenizer = BertTokenizer.from_pretrained(config['text_encoder_path'])
            text_encoder = BertModel(config=bert_config, add_pooling_layer=False)
            msg = text_encoder.load_state_dict(bert_state_dict, strict=False)
        else:
            raise NotImplementedError
        print('load bert weight: ', msg)
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

    def forward(self, frame_feats, frame_feats_mask, text_input_ids, text_segment_ids, text_attention_mask, alpha=0):
        batch_size = frame_feats.size()[0]
        loss_itc, loss_ima, loss_itm, loss_mlm, loss_mfm = 0, 0, 0, 0, 0
        loss_itm_pos = None
        sim_i2t, sim_t2i = torch.ones((batch_size, batch_size), device=frame_feats.device), \
                           torch.ones((batch_size, batch_size), device=frame_feats.device)

        # process text
        probability_matrix = torch.ones_like(text_input_ids, device=frame_feats.device) * self.mlm_probability

        # image encode
        frame_attn_mask = frame_feats_mask
        frame_emb = frame_feats
        if self.use_raw_image:
            frame_emb, frame_attn_mask = self.frame_encoder(frame_emb, frame_attn_mask)

        visual_emb, visual_attn_mask = self.visual_encoder(frame_emb, frame_attn_mask)

        # text encode
        text_emb = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask,
                                     token_type_ids=text_segment_ids)[0]
        text_attn_mask = text_attention_mask

        if 'itc' in self.config['tasks']:
            loss_itc, loss_ima, sim_i2t, sim_t2i = self.itc_forward(visual_emb, text_emb, frame_feats, frame_feats_mask,
                                                                    text_input_ids,
                                                                    text_attention_mask, text_segment_ids, alpha)

        if 'mlm' in self.config['tasks'] and 'mfm' in self.config['tasks']:
            loss_mlm, loss_mfm, loss_itm_pos = self.mlm_mfm_forward(frame_emb, frame_attn_mask, text_input_ids, text_attn_mask,
                                                                    text_segment_ids, probability_matrix, alpha)
        elif 'mlm' in self.config['tasks']:
            loss_mlm = self.mlm_forward(visual_emb, visual_attn_mask, text_input_ids, text_attn_mask, text_segment_ids,
                                        probability_matrix, alpha)
        elif 'mfm' in self.config['tasks']:
            loss_mfm = self.mfm_forward(frame_emb, frame_attn_mask, text_emb, text_attn_mask)

        if 'itm' in self.config['tasks']:
            loss_itm_neg = self.itm_forward(visual_emb, text_emb, visual_attn_mask, text_attn_mask, sim_i2t, sim_t2i)
            loss_itm += loss_itm_neg
            if loss_itm_pos is not None:
                loss_itm = (loss_itm_neg + loss_itm_pos) / 2

        return loss_itc, loss_ima, loss_itm, loss_mlm, loss_mfm

    def itc_forward(self, visual_emb, text_emb, frame_feats, frame_feats_mask, input_ids, attention_mask, segment_ids,
                    alpha):
        '''image text contrastive'''
        with torch.no_grad():
            self._momentum_update()
            visual_attn_mask_m = frame_feats_mask
            visual_emb_m = frame_feats
            if self.use_raw_image:
                visual_emb_m, visual_attn_mask_m = self.frame_encoder_m(frame_feats, frame_feats_mask)
            cls_visual_emb_m = self.visual_encoder_m(visual_emb_m, visual_attn_mask_m)[0][:, 0, :]
            cls_visual_emb_m = F.normalize(self.vision_proj_m(cls_visual_emb_m), dim=-1)
            cls_visual_emb_m_all = torch.cat([cls_visual_emb_m.t(), self.vision_queue.clone().detach()], dim=1)

            cls_text_emb_m = self.text_encoder_m(input_ids=input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=segment_ids)[0][:, 0, :]
            cls_text_emb_m = F.normalize(self.text_proj_m(cls_text_emb_m), dim=-1)
            cls_text_emb_m_all = torch.cat([cls_text_emb_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = cls_visual_emb_m @ cls_text_emb_m_all / self.temp
            sim_t2i_m = cls_text_emb_m @ cls_visual_emb_m_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(frame_feats.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        cls_visual_emb = F.normalize(self.vision_proj(visual_emb[:, 0, :]), dim=-1)
        cls_text_emb = F.normalize(self.text_proj(text_emb[:, 0, :]), dim=-1)
        # 计算图像文本之间的相似度
        sim_i2t = cls_visual_emb @ cls_text_emb_m_all / self.temp
        sim_t2i = cls_text_emb @ cls_visual_emb_m_all / self.temp
        # image text contrastive 的交叉熵loss
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_itc = (loss_i2t + loss_t2i) / 2

        # 模态内全局对比
        loss_ima = 0
        if 'ima' in self.config['tasks']:
            sim_i2i = cls_visual_emb @ cls_visual_emb_m_all / self.temp
            sim_t2t = cls_text_emb @ cls_text_emb_m_all / self.temp
            loss_ima1 = -torch.sum(F.log_softmax(sim_i2i.float(), dim=1) * sim_targets, dim=1).mean()
            loss_ima2 = - torch.sum(F.log_softmax(sim_t2t.float(), dim=1) * sim_targets, dim=1).mean()
            loss_ima = (loss_ima1 + loss_ima2) / 2

        if self.queue_size > 0:
            self._dequeue_and_enqueue(cls_visual_emb_m, cls_text_emb_m)

        return loss_itc, loss_ima, sim_i2t, sim_t2i

    def itm_forward(self, visual_emb, text_emb, visual_attn_mask, text_attn_mask, sim_i2t, sim_t2i):
        '''image text match'''
        bs = visual_emb.size()[0]
        if self.use_hard_negs:
            visual_emb_neg, visual_attn_mask_neg, text_emb_neg, text_attn_mask_neg = self.get_hard_negs(bs,
                                                                                                        sim_i2t,
                                                                                                        sim_t2i,
                                                                                                        visual_emb,
                                                                                                        text_emb,
                                                                                                        visual_attn_mask,
                                                                                                        text_attn_mask, )

            text_emb_all = torch.cat([text_emb_neg, text_emb], dim=0)
            text_attn_mask_all = torch.cat([text_attn_mask_neg, text_attn_mask], dim=0)
            visual_emb_all = torch.cat([visual_emb, visual_emb_neg], dim=0)
            visual_attn_mask_all = torch.cat([visual_attn_mask, visual_attn_mask_neg], dim=0)
            itm_labels = torch.zeros(2 * bs, dtype=torch.long, device=visual_emb.device)
        else:
            text_emb_all = text_emb
            text_attn_mask_all = text_attn_mask
            reverse_idx = torch.tensor(list(range(bs))[::-1], device=visual_emb.device)
            visual_emb_all = visual_emb[reverse_idx]
            visual_attn_mask_all = visual_attn_mask[reverse_idx]
            itm_labels = torch.zeros(bs, dtype=torch.long, device=visual_emb.device)

        if self.fusion == 'merge_attention':
            multimodal_emb = self.multimodal_encoder(
                hidden_states=torch.cat([visual_emb_all, text_emb_all], dim=1),
                attention_mask=torch.cat([visual_attn_mask_all, text_attn_mask_all], dim=1))[0]
            visual_cls_emb = multimodal_emb[:, 0, :]
            text_cls_emb = multimodal_emb[:, visual_emb.size()[1], :]
            if self.pooling != '':
                itm_multimodal_emb = self.emb_pool([visual_cls_emb, text_cls_emb], pooling=self.pooling)
            else:
                itm_multimodal_emb = text_cls_emb

        elif self.fusion == 'cross_attention':
            if self.cross_type == 'text':
                output = self.multimodal_encoder_text(
                    hidden_states=text_emb_all,
                    attention_mask=text_attn_mask_all,
                    encoder_hidden_states=visual_emb_all,
                    encoder_attention_mask=visual_attn_mask_all,
                    output_hidden_states=True)

                if self.pooling != '':
                    cls_emb_list = [output.hidden_states[-i][:, 0, :] for i in range(1, 5)]
                    itm_multimodal_emb = self.emb_pool(cls_emb_list, pooling=self.pooling)
                else:
                    itm_multimodal_emb = output.last_hidden_state[:, 0, :]

            elif self.cross_type == 'image':
                output = self.multimodal_encoder_image(
                    hidden_states=visual_emb_all,
                    attention_mask=visual_attn_mask_all,
                    encoder_hidden_states=text_emb_all,
                    encoder_attention_mask=text_attn_mask_all,
                    output_hidden_states=True)

                if self.pooling != '':
                    cls_emb_list = [output.hidden_states[-i][:, 0, :] for i in range(1, 5)]
                    itm_multimodal_emb = self.emb_pool(cls_emb_list, pooling=self.pooling)
                else:
                    itm_multimodal_emb = output.last_hidden_state[:, 0, :]
            else:
                output_text = self.multimodal_encoder_text(
                    hidden_states=text_emb_all,
                    attention_mask=text_attn_mask_all,
                    encoder_hidden_states=visual_emb_all,
                    encoder_attention_mask=visual_attn_mask_all,
                    output_hidden_states=True)
                output_image = self.multimodal_encoder_image(
                    hidden_states=visual_emb_all,
                    attention_mask=visual_attn_mask_all,
                    encoder_hidden_states=text_emb_all,
                    encoder_attention_mask=text_attn_mask_all,
                    output_hidden_states=True)

                if self.pooling != '':
                    cls_emb_text_list = [output_text.hidden_states[-i][:, 0, :] for i in range(1, 3)]
                    cls_emb_image_list = [output_image.hidden_states[-i][:, 0, :] for i in range(1, 3)]
                    multimodal_emb_text = self.emb_pool(cls_emb_text_list, pooling='max')
                    multimodal_emb_image = self.emb_pool(cls_emb_image_list, pooling='max')

                    #                     multimodal_emb_text = output_text.last_hidden_state[:, 0, :]
                    #                     multimodal_emb_image = output_image.last_hidden_state[:, 0, :]
                    itm_multimodal_emb = self.emb_pool([multimodal_emb_image, multimodal_emb_text],
                                                       pooling=self.pooling)
                else:
                    multimodal_emb_text = output_text.last_hidden_state[:, 0, :]
                    multimodal_emb_image = output_image.last_hidden_state[:, 0, :]
                    itm_multimodal_emb = torch.cat([multimodal_emb_image, multimodal_emb_text], dim=-1)
        else:
            raise NotImplementedError

        itm_output = self.itm_head(itm_multimodal_emb)
        loss_itm_neg = F.cross_entropy(itm_output, itm_labels)

        return loss_itm_neg

    def mlm_forward(self, visual_emb, visual_attn_mask, input_ids, text_attn_mask, segment_ids, probability_matrix,
                    alpha):
        '''mask language modeling'''
        mlm_input_ids = input_ids.clone()
        mlm_labels = input_ids.clone()
        mlm_input_ids, mlm_labels = self.mask_text(mlm_input_ids, self.vocab_size, input_ids.device, targets=mlm_labels,
                                                   probability_matrix=probability_matrix)

        mlm_text_emb = \
            self.text_encoder(input_ids=mlm_input_ids, attention_mask=text_attn_mask, token_type_ids=segment_ids)[0]

        if self.fusion == 'merge_attention':
            output = self.multimodal_encoder(
                hidden_states=torch.cat([visual_emb, mlm_text_emb], dim=1),
                attention_mask=torch.cat([visual_attn_mask, text_attn_mask], dim=1))
            mlm_multimodal_text_emb = output.last_hidden_state[:, visual_emb.size()[1]:, :]
        elif self.fusion == 'cross_attention':
            output = self.multimodal_encoder_text(
                hidden_states=mlm_text_emb,
                attention_mask=text_attn_mask,
                encoder_hidden_states=visual_emb,
                encoder_attention_mask=visual_attn_mask)
            mlm_multimodal_text_emb = output.last_hidden_state
        else:
            raise NotImplementedError

        loss_mlm = self.mlm_head(mlm_multimodal_text_emb,
                                 labels=mlm_labels,
                                 alpha=alpha
                                 ).loss
        return loss_mlm

    def mfm_forward(self, frame_feats, frame_feats_mask, text_emb, text_attn_mask, temp=0.07):
        '''mask frame modeling'''
        mfm_labels = frame_feats.clone()
        mfm_frame_feats, mfm_labels_index = self.mask_frame(frame_feats, frame_feats_mask)
        mfm_visual_emb, visual_attn_mask = self.visual_encoder(mfm_frame_feats, frame_feats_mask)
        # mfm_visual_emb, visual_attn_mask = mfm_frame_feats, frame_feats_mask

        if self.fusion == 'merge_attention':
            output = self.multimodal_encoder(
                hidden_states=torch.cat([mfm_visual_emb, text_emb], dim=1),
                attention_mask=torch.cat([visual_attn_mask, text_attn_mask], dim=1))
            mfm_multimodal_visual_emb = output.last_hidden_state[:, :mfm_visual_emb.size()[1], :]
        elif self.fusion == 'cross_attention':
            output = self.multimodal_encoder_image(
                hidden_states=mfm_visual_emb,
                attention_mask=visual_attn_mask,
                encoder_hidden_states=text_emb,
                encoder_attention_mask=text_attn_mask)
            mfm_multimodal_visual_emb = output.last_hidden_state
        else:
            raise NotImplementedError

        # 去掉cls emb
        if self.visual_cls:
            mfm_multimodal_visual_emb = mfm_multimodal_visual_emb[:, 1:, :]
        loss_mfm = self.mfm_head(mfm_multimodal_visual_emb, mfm_labels, mfm_labels_index, frame_feats_mask, temp)

        return loss_mfm

    def mlm_mfm_forward(self, frame_feats, frame_feats_mask, input_ids, text_attn_mask, segment_ids, probability_matrix,
                        alpha):
        '''mask language and frame modeling'''
        ### mask frame
        mfm_labels = frame_feats.clone()
        mfm_frame_feats, mfm_labels_index = self.mask_frame(frame_feats, frame_feats_mask)

        ### mask language
        mlm_input_ids = input_ids.clone()
        mlm_labels = input_ids.clone()
        mlm_input_ids, mlm_labels = self.mask_text(mlm_input_ids, self.vocab_size, input_ids.device, targets=mlm_labels,
                                                   probability_matrix=probability_matrix)

        ### text encode
        mlm_text_emb = \
            self.text_encoder(input_ids=mlm_input_ids, attention_mask=text_attn_mask, token_type_ids=segment_ids)[0]

        ### video encode
        mfm_visual_emb, visual_attn_mask = self.visual_encoder(mfm_frame_feats, frame_feats_mask)
        # mfm_visual_emb, visual_attn_mask = mfm_frame_feats, frame_feats_mask

        ### video text fusion
        if self.fusion == 'merge_attention':
            output = self.multimodal_encoder(
                hidden_states=torch.cat([mfm_visual_emb, mlm_text_emb], dim=1),
                attention_mask=torch.cat([visual_attn_mask, text_attn_mask], dim=1))
            mfm_multimodal_visual_emb = output.last_hidden_state[:, :mfm_visual_emb.size()[1], :]
            mlm_multimodal_text_emb = output.last_hidden_state[:, mfm_visual_emb.size()[1]:, :]
        elif self.fusion == 'cross_attention':
            output_text = self.multimodal_encoder_text(
                hidden_states=mlm_text_emb,
                attention_mask=text_attn_mask,
                encoder_hidden_states=mfm_visual_emb,
                encoder_attention_mask=visual_attn_mask)
            output_image = self.multimodal_encoder_image(
                hidden_states=mfm_visual_emb,
                attention_mask=visual_attn_mask,
                encoder_hidden_states=mlm_text_emb,
                encoder_attention_mask=text_attn_mask)
            mlm_multimodal_text_emb = output_text.last_hidden_state
            mfm_multimodal_visual_emb = output_image.last_hidden_state
        else:
            raise NotImplementedError

        loss_mlm = self.mlm_head(mlm_multimodal_text_emb,
                                 labels=mlm_labels,
                                 alpha=alpha
                                 ).loss

        loss_itm_pos = None
        # itm loss positive samples
        if 'itm' in self.config['tasks']:
            multimodal_emb_text = mlm_multimodal_text_emb[:, 0, :]
            multimodal_emb_image = mfm_multimodal_visual_emb[:, 0, :]
            itm_multimodal_emb = torch.cat([multimodal_emb_image, multimodal_emb_text], dim=-1)
            itm_output = self.itm_head(itm_multimodal_emb)
            itm_labels = torch.ones(frame_feats.size()[0], dtype=torch.long, device=frame_feats.device)
            loss_itm_pos = F.cross_entropy(itm_output, itm_labels)

        # 去掉cls emb
        if self.visual_cls:
            mfm_multimodal_visual_emb = mfm_multimodal_visual_emb[:, 1:, :]
        loss_mfm = self.mfm_head(mfm_multimodal_visual_emb, mfm_labels, mfm_labels_index, frame_feats_mask, normalize=True)

        return loss_mlm, loss_mfm, loss_itm_pos

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, visual_emb, text_emb, idx=None):
        # gather keys before updating queue
        visual_emb = concat_all_gather(visual_emb) if dist.is_initialized() else visual_emb
        text_emb = concat_all_gather(text_emb) if dist.is_initialized() else text_emb
        batch_size = visual_emb.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.vision_queue[:, ptr:ptr + batch_size] = visual_emb.T
        self.text_queue[:, ptr:ptr + batch_size] = text_emb.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    def get_hard_negs(self, bs, sim_i2t, sim_t2i, visual_emb, text_emb, visual_attn_mask, text_attn_mask):
        with torch.no_grad():
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)
            weights_i2t.fill_diagonal_(0)  # 避免选取负样本的时候选到自己对应的文本
            weights_t2i.fill_diagonal_(0)

        # torch.clamp(weights_t2i, min=0)
        # torch.clamp(weights_i2t, min=0)

        # select a negative image for each text
        visual_emb_neg = []
        visual_attn_mask_neg = []
        for b in range(bs):
            # 对weights_t2i[b]进行采样，也就是根据第b个文本和当前batch内图像的相似度进行采样，相似度越高的才采样的几率越大
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            visual_emb_neg.append(visual_emb[neg_idx])
            visual_attn_mask_neg.append(visual_attn_mask[neg_idx])
        visual_emb_neg = torch.stack(visual_emb_neg, dim=0)
        visual_attn_mask_neg = torch.stack(visual_attn_mask_neg, dim=0)

        # select a negative text for each image
        text_emb_neg = []
        text_attn_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_emb_neg.append(text_emb[neg_idx])
            text_attn_mask_neg.append(text_attn_mask[neg_idx])
        text_emb_neg = torch.stack(text_emb_neg, dim=0)
        text_attn_mask_neg = torch.stack(text_attn_mask_neg, dim=0)

        return visual_emb_neg, visual_attn_mask_neg, text_emb_neg, text_attn_mask_neg

    def mask_text(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            # 根据probability_matrix中的概率来确定该位置是否mask， masked_indices中为True的就是需要mask
            masked_indices = torch.bernoulli(probability_matrix).bool()

        # padding和文本头部的cls token不需要进行mask
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        masked_indices[input_ids == self.tokenizer.sep_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool().to(device) & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool().to(
            device) & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def mask_frame(self, visual_emb, visual_attn_mask):
        probability_matrix = torch.full(visual_emb.size()[:2], self.mfm_probability, device=visual_emb.device)
        probability_matrix = probability_matrix * visual_attn_mask

        masked_indices = torch.bernoulli(probability_matrix).bool()
        if self.visual_cls:
            masked_indices[:, 0] = False

        video_labels_index = torch.arange(visual_emb.size(0) * visual_emb.size(1), device=visual_emb.device).view(-1,
                                                                                                                  visual_emb.size(
                                                                                                                      1))
        video_labels_index = -100 * ~masked_indices + video_labels_index * masked_indices

        masked_indices_unsqueeze = masked_indices.unsqueeze(-1).expand_as(visual_emb)
        inputs = visual_emb.data.masked_fill(masked_indices_unsqueeze, 0.0)

        return inputs, video_labels_index

    def emb_pool(self, emb_list, pooling=''):
        if pooling == 'max':
            pool_emb = torch.max(torch.stack(emb_list, dim=1), dim=1)[0]
        elif pooling == 'mean':
            pool_emb = torch.mean(torch.stack(emb_list, dim=1), dim=1)
        elif pooling == 'weight':
            emb = torch.stack(emb_list, dim=1)
            emb_w = torch.softmax(self.fusion_layer(emb), dim=1)
            pool_emb = torch.sum(emb * emb_w, dim=1)
        elif pooling == 'enhance':
            pool_emb = self.fusion_layer(emb_list)
        else:
            raise ValueError
        return pool_emb


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    config = {'text_encoder_arch': 'bert', 'text_encoder_path': '../pretrained_model/roberta/',

              'frame_encoder_arch': 'clip_vit',
              'frame_encoder_config_path': '../pretrained_model/clip_base_32/config.json',
              'frame_encoder_path': '../pretrained_model/clip_base_32/pytorch_model.bin',
              'frame_emb_type': 'frame',

              'use_visual_encoder': True,
              'visual_encoder_arch': 'transformer_prenorm',
              'visual_encoder_config_path': '../configs/visual_encoder_config_prenorm.json',
              'visual_encoder_path': '',

              'multimodal_config_path': '../configs/cross_attention_config.json', 'text_embed_dim': 768,
              'visual_embed_dim': 1024, 'max_title_len': 15, 'max_asr_len': 25,
              'max_ocr_len': 15,
              'use_multimodal_enhance': False, 'use_ocr': True, 'use_prompt': True,
              'fusion': 'cross_attention', 'mm_embed_dim': 768, 'max_len': 512, 'label2_nums': 1000, 'use_asr': True,
              'cross_type': 'image_text', 'queue_size': 0, 'momentum': 0.995, 'temp': 0.07,
              'use_tfidf': False, 'mlm_probability': 0.15, 'title_mlm_probability': 0.25, 'asr_mlm_probability': 0.15,
              'ocr_mlm_probability': 0.1, 'tasks': 'itc,itm,mfm,mlm', 'mfm_probability': 0.15, 'pooling': '',
              'use_raw_image': False, 'use_hard_negs': False}
    model = MultiModal_Pretrain(config)

    image_feats = torch.randn((4, 3, 768))
    image_masks = torch.ones((4, 3))

    text_input_ids = torch.randint(0, 100, (4, 274))
    text_segment_ids = torch.randint(0, 2, (4, 274))
    text_attention_mask = torch.randint(0, 2, (4, 274))

    output = model(image_feats, image_masks, text_input_ids, text_segment_ids, text_attention_mask)
    print(output)
