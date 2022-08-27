import os
import io
import json
import torch
import zipfile
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from models.backbone.clip_vit import CLIPVisionModel
from models.backbone import swin
from transformers.models.clip.configuration_clip import CLIPVisionConfig


class RawFrameDataset(Dataset):

    def __init__(self,
                 ann_path: str,
                 zip_frame_dir: str,
                 max_video_frames: int = 32):
        """ This class is used to load raw video frames.
        Args:
            ann_paths (str): the annotation file path.
            zip_frame_dir (str): the directory that saves zip frames.
            max_video_frames (str): the maximum number of video frames.
        """
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        self.zip_frame_dir = zip_frame_dir
        self.max_video_frames = max_video_frames
        
        # we follow the common practice as in the ImageNet's preprocessing.
        self.transform = Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> dict:
        return len(self.anns)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Extract the frame tensor from zipped file.
        The output tensor is in shape of [MAX_FRAMES, 3, 224, 224]
        """
        feedid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, feedid[-3:], f'{feedid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        img_name_list = handler.namelist()
        img_name_list = sorted(img_name_list)
        img_name_list = img_name_list[:self.max_video_frames]
        img_tensor = torch.zeros(self.max_video_frames, 3, 224, 224)
        for i, img_name in enumerate(img_name_list):
            i_img_content = handler.read(img_name)
            i_img = Image.open(io.BytesIO(i_img_content))
            i_img_tensor = self.transform(i_img)
            img_tensor[i, ...] = i_img_tensor
        handler.close()
        num_frames = torch.LongTensor([len(img_name_list)])
        return dict(img=img_tensor, num_frames=num_frames)


def parse_args():
    parser = argparse.ArgumentParser("Visual feature extraction")
    parser.add_argument('--zip_frame_dir', type=str, default='/home/tione/notebook/data/zip_frames/labeled/')
    parser.add_argument('--ann_path', type=str, default='/home/tione/notebook/data/annotations/labeled.json')
    parser.add_argument('--swin_pretrained', type=str, default='opensource_models/swin_tiny_patch4_window7_224.pth')
    parser.add_argument('--output_path', type=str, default='/home/tione/notebook/data/zip_feats/labeled.zip')
    parser.add_argument('--frame_encoder_path', type=str, default='./pretrained_model/vit_base/')
    parser.add_argument('--frame_encoder_arch', type=str, default='swin')
    parser.add_argument('--frame_encoder_config_path', type=str, default='./configs/visual_encoder_config.json')
    args = parser.parse_args()
    return args


def build_model(args) -> torch.nn.Module:
    """ Load the pretrianed feature extractor (Swin-T here). """
    # if not os.path.isfile(swin_pretrained):
    #     raise IOError(f"Cannot load pretrained swin model from {swin_pretrained}."
    #                   "Please manually download it from https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth")
    # model = swin.swin_tiny(swin_pretrained)
    
    clip_config = CLIPVisionConfig.from_pretrained(args.frame_encoder_config_path)
    model = CLIPVisionModel(clip_config)
    state_dict = torch.load(args.frame_encoder_path, map_location='cpu')
    vit_state_dict = {}
    for k, v in state_dict.items():
        if 'text_model' in k:
            continue
        vit_state_dict[k] = v
    msg = model.load_state_dict(vit_state_dict, strict=False)
    print('clip vit: ', msg)
    
    if torch.cuda.is_available():
        model = DataParallel(model.cuda(), device_ids=list(range(torch.cuda.device_count())))
    model.eval()
    return model


def main():
    args = parse_args()
    model = build_model(args)

    dataset = RawFrameDataset(args.ann_path, args.zip_frame_dir)
    # batch-size == 8 is fine for V100 GPU, please consider use smaller batch-size if OOM issue occurs.
    dataloader = DataLoader(dataset, batch_size=8, num_workers=24, shuffle=False, pin_memory=True, drop_last=False)

    assert not os.path.isfile(args.output_path), f"{args.output_path} already exists. " \
                                                  "If you want to override it, please manually delete this file."
    output_handler = zipfile.ZipFile(args.output_path, 'w', compression=zipfile.ZIP_STORED)

    with torch.no_grad():
        cur = 0
        for dataitem in dataloader:
            img, num_frames = dataitem['img'], dataitem['num_frames']
            B, L = img.shape[0:2]
            img = img.view((B * L, ) + img.shape[2:])
            feature, _ = model(img)
            feature = feature.view(B, L, -1)
            feature = feature.cpu().numpy().astype(np.float16)
            for i in range(B):
                feedid = dataset.anns[cur]['id']
                ioproxy = io.BytesIO()
                np.save(ioproxy, feature[i, :int(num_frames[i])])
                npy_str = ioproxy.getvalue()
                output_handler.writestr(f'{feedid}.npy', npy_str)
                cur += 1
                if cur % 1000 == 0:
                    print(f"Extract feature {cur}/{len(dataset)}")
    output_handler.close()


if __name__ == '__main__':
    main()
