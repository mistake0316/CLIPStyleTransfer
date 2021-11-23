import torch
import clip

class CLIPLoss(torch.nn.Module):

    def __init__(self, img_size, device="cuda"):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=img_size//32)
        self.norm = self.preprocess.transforms[-1]

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        self.norm(image)
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity