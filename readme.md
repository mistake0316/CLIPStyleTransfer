# [CLIP Style Transfer](https://github.com/mistake0316/CLIPStyleTransfer)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mistake0316/CLIPStyleTransfer/blob/main/Optimization_Experiment.ipynb) : Standard Approach 
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mistake0316/CLIPStyleTransfer/blob/main/Optimization_Augmentation_Experiment.ipynb) : Augmentation Approach (More Prefer)
* [Open In Replicate](https://replicate.com/mistake0316/style-transfer-clip) : Augmentation Approach

In this repo, you can control image's style with text prompts, such as ```(PIL.Image("doge.jpg"), "cheese cake")```.<br>
I apply CLIP Loss on style transfer model's style code, then we can cost about 1 min(with gpu) to get folllowing results:

![doge](https://replicate.com/api/models/mistake0316/style-transfer-clip/files/0f51aae9-ef91-4ee6-a83f-065abbb7a65c/doge.jpeg)

----
## cheese cake
![cheese cake](https://replicate.com/api/models/mistake0316/style-transfer-clip/files/87e8ca5b-a313-4a74-83a8-e59fd8d444db/out.png)

## green and blue mosaic
![green and blue mosaic](https://replicate.com/api/models/mistake0316/style-transfer-clip/files/b6ad0e9a-5388-4364-8acd-2d0a4f5cede1/out.png)

## bush
![bush](https://replicate.com/api/models/mistake0316/style-transfer-clip/files/30280404-ecbf-4297-a6ed-284c9b9d3cec/out.png)

## Leopard
![Leopard](https://replicate.com/api/models/mistake0316/style-transfer-clip/files/e8ef20e6-8952-4d6d-aa53-39ecf5151735/out.png)

## firewor
![firework](https://replicate.com/api/models/mistake0316/style-transfer-clip/files/4fb41829-6d3e-4bd2-9fac-4264b53e2d0e/out.png)

## bubble tea
![bubble tea](https://replicate.com/api/models/mistake0316/style-transfer-clip/files/3f64353b-40b8-472b-a927-4d879ca402be/out.png)

---
# Experiments Describe
![style transfer model](https://d3i71xaburhd42.cloudfront.net/7821cfd68b0b67e3c20dcbc82a71e77af9e09931/3-Figure2-1.png)
> Image is in [Exploring the structure of a real-time, arbitrary neural artistic stylization network](https://arxiv.org/abs/1705.06830)

In this repo, we edit the style code $\overrightarrow{S}$ with CLIP Loss : ```CLIP_model(T(c, S), text_prompt)```.

For ***Augmentation Approach***, we augment $T(c, \overrightarrow{S})$ before compute CLIP loss.

I didn't upload some bad result implementation such as
* Multiple Content image ```c```
* Noise Style Code ```S_noise = S * gaussian(mean=1, std=eplison)```

---
# Acknowledgments
I borrow 
* Some codes from [StyleCLIP](https://github.com/orpatashnik/StyleCLIP), [style-augmentation](https://github.com/philipjackson/style-augmentation)
* Base model : [magenta arbitrary style transfer](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2), [CLIP](https://github.com/openai/CLIP)
* Idea of augmentation after generate an image : [CLIPDraw](https://arxiv.org/abs/2106.14843)

