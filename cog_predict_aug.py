# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import cog
import torch
import sys

#@title import
import torch
import torchvision.transforms as T
from utils.stylePredictor26 import pretrainedStylePredictor
from utils.transform26 import pretrainedGhiasi

import clip
from utils.clip_loss import CLIPLoss
from PIL import Image

import torchvision.transforms as T
from tqdm.auto import tqdm

import numpy as np


from pathlib import Path
import tempfile


class Predictor(cog.Predictor):
    def setup(self):
      """Load the model into memory to make running multiple predictions efficient"""

      #@title load models
      print("loading models")
      self.device = device = ["cpu", "cuda"][torch.cuda.is_available()]
      print(f"device : {device}")
      self.SP = pretrainedStylePredictor().to(device).eval()
      self.G = pretrainedGhiasi().to(device).eval()
      self.clip_loss = CLIPLoss(384, device).eval()
      print("done")
            
    @cog.input("image", type=Path, help="input image")
    @cog.input("text", type=str, help="prompt text")
    @cog.input("optimize_steps", type=int, default=100, help="total steps")
    @cog.input("display_every_step", type=int, default=10, help="")
    @cog.input("resize_flag", type=bool, default=True, options=[True, False], help="resize image or not")
    @cog.input("short_edge_target_len", type=int, default=384, help="the short edge size if resize_flag is True")
    @cog.input("center_crop_flag", type=bool, default=True, options=[True, False], help="center crop content image or not")
    @cog.input("output_aug_flag", type=bool, default=True, options=[True, False], help="augment before compute clip loss")
    @cog.input("output_aug_time", type=int, default=8, help="augment times if output_aug_flag is True")
#     @cog.input("style_code_aug_flag", type=bool, default=True, options=[True, False], help="resize image or not")
#     @cog.input("style_code_aug_gaussian_noise_std", type=float, default=0.01,
#                help="feed_code = code * (1+noise_std*torch.randn_like(code))")
    @cog.input("lr", type=float, default=0.01)
    def predict(
      self,
      image, text,
      optimize_steps, display_every_step,
      resize_flag, short_edge_target_len,
      center_crop_flag,
      output_aug_flag, output_aug_time,
#       style_code_aug_flag, style_code_aug_gaussian_noise_std,
      lr,
    ):
      """Run a single prediction on the model"""
      device = self.device
      out_path = Path(tempfile.mkdtemp()) / "out.png"
      
      SP = self.SP
      G = self.G
      clip_loss = self.clip_loss
      
      print("loading content")
      content = Image.open(image).convert('RGB')
      
      style_preprocess = torch.nn.Sequential(
        T.Resize(256,),
        T.CenterCrop(256),
        torch.nn.AvgPool2d(kernel_size=3,stride=1)
      )
      
      short_edge_len = int(np.ceil(min(content.size)/128)*128) if not resize_flag else 384
      clip_loss.avg_pool.kernel_size = clip_loss.avg_pool.stride = short_edge_len//32
      
      content_preprocess = torch.nn.Sequential()
      center_crop = T.CenterCrop(short_edge_len)
      content_preprocess.add_module("resize", T.Resize(short_edge_len,))
      if center_crop_flag:
        content_preprocess.add_module("center_crop",center_crop,)
      
      Aug = torch.nn.Sequential(
        T.RandomAffine(
          degrees=180,
          shear=5,
          scale=(1, 1.5),
        ),
        T.ColorJitter(
          brightness=0.1,
          contrast=0.1,
          saturation=0.1,
          hue=0.1,
        ),
        T.RandomCrop(short_edge_target_len),
      )

      aug_fun = lambda img: torch.cat([Aug(img) for _ in range(output_aug_time)])
      
      #@title reconstruct with Ghiasi's style transfer model
      content_tensor = T.ToTensor()(content)
      content_tensor = content_preprocess(content_tensor)

      style_tensor = T.ToTensor()(content)
      style_tensor = style_preprocess(style_tensor)

      content_tensor = content_tensor.unsqueeze(0).to(device)
      style_tensor = style_tensor.unsqueeze(0).to(device)
      with torch.no_grad():
        initial_style_code = SP(style_tensor)
      
      
      def tensor_to_image_file(tensor, path):
        array = (255*tensor.detach().squeeze().cpu().permute(1,2,0).numpy()).astype(np.uint8)
        Image.fromarray(array).save(path)
        return path
      
      print(text)
      tokens = clip.tokenize([text]).to(device)
      print(tokens)
      code = initial_style_code.clone()
      code.requires_grad_()
      optimizer = torch.optim.Adam([code],lr=lr)
      
      def compute_step():
#         if style_code_aug_flag:
#           feed_code = code * (1 + style_code_aug_gaussian_noise_std*torch.randn_like(code))
#         else:
#           feed_code = code
        feed_code = code
        
        result = G(content_tensor, feed_code)
        if output_aug_flag:
          aug_result = aug_fun(result)
        else:
          aug_result = result
        
        c_loss = torch.mean(clip_loss(center_crop(aug_result), tokens))

        loss = c_loss
        return loss, result
    
      for idx in range(optimize_steps):
        loss, result = compute_step()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if idx % display_every_step == 0 or idx == optimize_steps-1:
          tensor_to_image_file(result[0], out_path)
          print(f"step : {idx}")
          print("\tloss : ", loss.item())
          yield out_path
      
