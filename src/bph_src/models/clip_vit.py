import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
from transformers.models.clip.configuration_clip import CLIPVisionConfig


class CLIPVisionModel(nn.Module):
    def __init__(self, config):
        super(CLIPVisionModel, self).__init__()
        self.vision_model = CLIPVisionTransformer(config)

    def get_input_embeddings(self):
        return self.vision_model.embeddings.patch_embedding

    def frozen_pooler_layer(self):
        self.vision_model.post_layernorm.requires_grad_(False)

    def forward(self, x, mask=None, emb_type='frame'):
        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)

        output = self.vision_model(pixel_values=x, return_dict=True)

        if emb_type == 'frame':
            pooler_output = output.pooler_output
            output_emb = pooler_output.view(B, N, -1)
        else:
            last_hidden_state = output.last_hidden_state
            seq_len = last_hidden_state.size()[1]
            output_emb = last_hidden_state.view(B, N*seq_len, -1)
            mask = mask.unsqueeze(-1).repeat(1, 1, seq_len).view(B, N*seq_len)

        return output_emb, mask


if __name__ == '__main__':
    config = CLIPVisionConfig.from_pretrained('../../pretrained_model/clip_base_32/config.json')
    model = CLIPVisionModel(config)

    state_dict = torch.load('../../pretrained_model/clip_base_32/pytorch_model.bin')
    vit_state_dict = {}
    for k, v in state_dict.items():
        if 'text_model' in k:
            continue
        vit_state_dict[k] = v
    msg = model.load_state_dict(vit_state_dict, strict=False)
    print(msg)

    x = torch.randn((2, 10, 3, 224, 224))
    mask = torch.ones((2, 10))
    output = model(x, mask,  emb_type='patch')
    print(output[0].shape)

    model.frozen_pooler_layer()
    for n, v in model.named_parameters():
        print(n, v.requires_grad)
