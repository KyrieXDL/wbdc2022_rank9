import math
import torch
import torch.utils.checkpoint
import math
import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.vit.modeling_vit import ViTSelfOutput, ViTIntermediate, ViTOutput
from transformers.models.bert.modeling_bert import BertLayer, BaseModelOutputWithPastAndCrossAttentions


class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # print(type(attention_probs), type(value_layer))
        # import pdb
        # pdb.set_trace()
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class ViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        return layer_output


class VisualEncoder_Prenorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #         self.embeddings = ViTEmbeddings(config)
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
        pos_num = config.pos_num
        if config.add_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            pos_num += 1
        if config.add_pos:
            self.position_embeddings = nn.Parameter(torch.zeros(1, pos_num, config.hidden_size))
            nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:

            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_hidden_states=False,
    ):
        if self.config.add_cls:
            batch_size = hidden_states.size()[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)

            cls_mask = torch.ones((hidden_states.size()[0], 1), device=hidden_states.device)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        if self.config.add_pos:
            N = hidden_states.size()[1]
            pos_embed = self.position_embeddings[:, :N, :] if self.config.add_cls else self.position_embeddings[:, 1:N + 1, :]
            hidden_states += pos_embed
        output_mask = attention_mask.clone()
        attention_mask = self.get_extended_attention_mask(attention_mask, attention_mask.size())
        attention_mask = attention_mask.to(hidden_states.device)

        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

        return hidden_states, output_mask


class VisualEncoder_Postnorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        pos_num = config.pos_num
        if config.add_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            pos_num += 1
        if config.add_pos:
            self.position_embeddings = nn.Parameter(torch.zeros(1, pos_num, config.hidden_size))
            nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:

            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError
        extended_attention_mask = extended_attention_mask  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_hidden_states=False,
    ):
        if self.config.add_cls:
            batch_size = hidden_states.size()[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)

            cls_mask = torch.ones((hidden_states.size()[0], 1), device=hidden_states.device)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        if self.config.add_pos:
            N = hidden_states.size()[1]
            pos_embed = self.position_embeddings[:, :N, :] if self.config.add_cls else self.position_embeddings[:, 1:N + 1, :]
            hidden_states += pos_embed

        output_mask = attention_mask.clone()
        all_hidden_states = () if output_hidden_states else None
        attention_mask = self.get_extended_attention_mask(attention_mask)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask
            )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, output_mask
