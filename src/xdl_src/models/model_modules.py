import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, MaskedLMOutput
from itertools import repeat
from transformers.models.bert.modeling_bert import ACT2FN


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio=8, dropout=0.3):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

        self.enhance = SENet(channels=output_size, ratio=8)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class MLMHead(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def forward(
            self,
            fusion_embed=None,
            labels=None,
            soft_labels=None,
            alpha=0,
            return_logits=False,
    ):

        sequence_output = fusion_embed
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if soft_labels is not None:
            loss_distill = -torch.sum(F.log_softmax(prediction_scores, dim=1) * soft_labels, dim=-1)
            loss_distill = loss_distill[labels != -100].mean()
            loss_distill = 0 if loss_distill != loss_distill else loss_distill
            masked_lm_loss = (1 - alpha) * masked_lm_loss + alpha * loss_distill

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
        )


class MFMHead(nn.Module):
    def __init__(self):
        super(MFMHead, self).__init__()
        self.block = MFMBlock(768)

    def forward(
            self,
            mfm_emb=None,
            mfm_labels=None,
            mfm_labels_index=None,
            visual_attn_mask=None,
            temp=1.0,
            normalize=False
    ):
        mfm_emb = self.block(mfm_emb)
        if normalize:
            mfm_emb = F.normalize(mfm_emb, p=2, dim=-1)
            mfm_labels = F.normalize(mfm_labels, p=2, dim=-1)
        # else:
        #     mfm_emb = self.block(mfm_emb)
            
        b, n, d = mfm_emb.shape
        mfm_emb = mfm_emb.reshape(b*n, d)
        mfm_labels = mfm_labels.reshape(b*n, d)

        logits_matrix = mfm_emb @ mfm_labels.t()
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = visual_attn_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e4

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        loss_mfm = -logpt

        video_labels_index_mask = (mfm_labels_index != -100)
        loss_mfm = loss_mfm.masked_select(video_labels_index_mask.view(-1))
        loss_mfm = loss_mfm.mean()

        return loss_mfm
    
    
class MFMBlock(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(embed_dim, embed_dim), ACT2FN['gelu'])
        self.LayerNorm = nn.LayerNorm(embed_dim, eps=1e-12)

        self.decoder = nn.Linear(embed_dim, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))
        self.decoder.bias = self.bias

    def forward(self, sequence_output):
        hidden_states = self.dense(sequence_output)
        hidden_states = self.LayerNorm(hidden_states)

        hidden_states = self.decoder(hidden_states)
        return hidden_states


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