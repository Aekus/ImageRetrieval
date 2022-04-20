import torch
import clip
from utils import load_image


class BaseModel():
    def __init__(self, device, paths, batchsize):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
        self.paths = paths
        self.batchsize = batchsize

    def image_to_tensor(self, path):
        im = load_image(path)
        return self.preprocess(im).to(self.device)

    def image_list_to_tensor(self, paths):
        preprocessed_images = [self.image_to_tensor(path) for path in paths]

        return torch.stack(preprocessed_images).to(self.device)

    def loss(self, im1, im2):
        im1_features = self.model.encode_image(self.preprocess(im1).unsqueeze(0).to(self.device))
        im2_features = self.model.encode_image(self.preprocess(im2).unsqueeze(0).to(self.device))

        loss = torch.norm(im1_features - im2_features)

        return loss