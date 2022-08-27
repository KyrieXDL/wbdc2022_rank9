

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, '../'))


from dataset.masklm import MaskLM, MaskVideo, ShuffleVideo
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from models.efficientNet import EfficientNet
from models.clip_vit import CLIPVisionModel
from transformers.models.clip.configuration_clip import CLIPVisionConfig

# torch.hub.set_dir(os.path.join(base_dir, '../../../data/cache'))
class WXModel(nn.Module):
    def __init__(self, args, cfg, task=['mlm', 'mfm'], init_from_pretrain=True):
        super().__init__()

        model_path = args.bert_dir
        self.end2end = args.end2end
        self.frame_emb_type = args.frame_emb_type
        #backbone to extract imgs features
        self.visual_backbone = self.build_visual_encoder(args)
        # self.visual_backbone=van_small(num_classes=768)
        # self.visual_backbone2=EfficientNet.from_pretrained('efficientnet-b0',num_classes=768,image_size=args.input_shape,dropout_rate=args.dropout)
        # bertconfig
        bert_cfg = BertConfig.from_pretrained(args.bert_dir)
        # uni_bert_cfg.num_hidden_layers = 1

        self.newfc_hidden = torch.nn.Linear(bert_cfg.hidden_size, cfg['HIDDEN_SIZE'])

        self.task = set(task)

        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=model_path)
            self.num_class = cfg['NUM_CLASSES']
            self.vocab_size = bert_cfg.vocab_size

        if 'mfm' in task:
            self.vm = MaskVideo()
            self.bert_mvm_lm_header = VisualOnlyMLMHead(bert_cfg)
        if 'mae' in task:
            self.mae=mae_vit_base_patch16_dec512d8b()
            
        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(bert_cfg.hidden_size, 1)

        if init_from_pretrain:
            self.bert = UniBertForMaskedLM.from_pretrained(model_path, config=bert_cfg)
        else:
            # Ԥѵ��
            self.bert = UniBertForMaskedLM(bert_cfg)
            
    def build_visual_encoder(self, args):
        if args.end2end:
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
    

    def forward(self, video_feature, video_mask, text_input_ids, text_mask, task=None):
        loss, pred = 0, None
        # if self.end2end and 'mfm' not in sample_task:
        #     video_feature, video_mask = self.visual_backbone(video_feature, video_mask, emb_type=self.frame_emb_type)
        
        if task is None:
            sample_task = self.task
        elif type(task) == str:
            sample_task = [task]
        elif type(task) == list:
            sample_task = task

        if 'mae' in sample_task:
            mae_loss,mae_pred,mae_mask=self.mae(video_feature)
            loss+=mae_loss
            # print('mae shape',mae_pred.shape)

        
        # video_feature = self.visual_backbone2(video_feature) #bs,32,768

        # perpare for pretrain task [mask and get task label]: {'mlm': mask title token} {'mfm': mask frame} {'itm': shuffle title-video}
        return_mlm = False
        if 'mlm' in sample_task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device)  # [SEP] �� MASK ��ʦ [SEP]
            return_mlm = True

        if 'mfm' in sample_task:
            if self.end2end:
                video_feature, video_mask = self.visual_backbone(video_feature, video_mask, emb_type=self.frame_emb_type)
            vm_input = video_feature
            input_feature, video_label = self.vm.torch_mask_frames(video_feature.cpu(), video_mask.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_label = video_label.to(video_feature.device)
            
            

        if 'itm' in sample_task:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)


        # concat features
        features, lm_prediction_scores, text_emb, video_emb = self.bert(video_feature, video_mask, text_input_ids, text_mask, return_mlm=return_mlm)
        
        features_mean = torch.mean(features, 1)
        embedding = self.newfc_hidden(features_mean)
        # embedding = self.newfc_hidden(features[:, 0, :])

        normed_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        dic_loss = {}
        # compute pretrain task loss
        if 'mlm' in sample_task:
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))
            # loss += masked_lm_loss / 1.25 / len(sample_task)
            loss += masked_lm_loss / len(sample_task)
            dic_loss['mlm'] = masked_lm_loss

        if 'mfm' in sample_task:
            vm_output = self.bert_mvm_lm_header(features[:, 1:video_feature.size()[1] + 1, :])
            masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input,
                                                     video_mask, video_label, normalize=False)
            masked_vm_loss = masked_vm_loss / 3
            loss += masked_vm_loss / len(sample_task)
            dic_loss['mfm'] = masked_vm_loss

        if 'itm' in sample_task:
            pred = self.newfc_itm(features[:, 0, :])
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
            # loss += itm_loss / 100 / len(sample_task)
            loss += itm_loss / len(sample_task)
            dic_loss['itm'] = itm_loss
        if 'itc' in sample_task:
            # text_emb, video_emb
            itc_loss = self.itc_loss(torch.mean(text_emb, dim=1), torch.mean(video_emb, dim=1))
            loss += itc_loss / len(sample_task)
            dic_loss['itc'] = itc_loss

        return (pred, normed_embedding, loss, dic_loss)
    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        if x.dim() == 5:
            # support multi-frame inputs
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            output_shape = (B, N, -1)
        else:
            output_shape = (x.shape[0], -1)
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    # calc mfm loss
    def calculate_mfm_loss(self, video_feature_output, video_feature_input,
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

        video_tr = video_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e4

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss
    
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


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = UniBert(config)
        self.cls = BertOnlyMLMHead(config)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None, return_mlm=False):
        encoder_outputs, text_emb, video_emb = self.bert(video_feature, video_mask, text_input_ids, text_mask)
        if return_mlm:
            return encoder_outputs, self.cls(encoder_outputs)[:, 1 + video_feature.size()[1]:, :], text_emb, video_emb
        else:
            return encoder_outputs, None


class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.video_fc = torch.nn.Linear(768, config.hidden_size)
        self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None):
        text_emb = self.embeddings(input_ids=text_input_ids)

        # text input is [CLS][SEP] t e x t [SEP]
        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]

        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]

        # reduce frame feature dimensions : 1536 -> 1024
        video_feature = self.video_fc(video_feature)
        video_emb = self.video_embeddings(inputs_embeds=video_feature)

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, video_emb, text_emb], 1)

        mask = torch.cat([cls_mask, video_mask, text_mask], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
        return encoder_outputs, text_emb, video_emb
