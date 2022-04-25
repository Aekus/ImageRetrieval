import torch
import clip
import numpy as np
from tqdm import tqdm
from utils import load_image, write_dict
from itertools import zip_longest


class BaseModel():
    def __init__(self, **args):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = args["device"]
        self.paths = args["paths"]
        self.batchsize = args["batchsize"]
        self.pred_outpath = args["pred_outpath"]

    def image_to_tensor(self, path):
        with torch.no_grad():
            im = load_image(path)
            return self.preprocess(im).to(self.device)

    def image_list_to_tensor(self, paths):
        preprocessed_images = [self.image_to_tensor(path) for path in paths]

        return torch.stack(preprocessed_images).to(self.device)

    def loss(self, im1, im2):
        with torch.no_grad():
            im1_features = self.model.encode_image(self.preprocess(im1).unsqueeze(0).to(self.device))
            im2_features = self.model.encode_image(self.preprocess(im2).unsqueeze(0).to(self.device))

            loss = torch.norm(im1_features - im2_features)

            return loss.item()

    def predict(self, feedbacks):
        return None, None

    def eval(self, annotations, write_predictions=True):
        predictions = {"predictions": []}
        for annot in tqdm(annotations["annotations"]):
            path = annot["source"]
            feedbacks = annot["feedbacks"]
            predictions["predictions"].append({"id": annot["id"],
                                               "source": path,
                                               "losses": [],
                                               "predicted outputs": [],
                                               "clip scores": []})
            prediction = predictions["predictions"][-1]

            for i in range(1, len(feedbacks) + 1):
                pred_path, score = self.predict(feedbacks[:i])
                prediction["predicted outputs"].append(path)
                prediction["clip scores"].append(score)

                loss = self.loss(load_image(path), load_image(pred_path))
                prediction["losses"].append(loss)

        losses = [prediction["losses"] for prediction in predictions["predictions"]]

        losses_t = [list(filter(None, i)) for i in zip_longest(*losses)]

        mean_losses = []
        for i in range(len(losses_t)):
            mean_losses.append(np.mean(losses_t[i]))

        if write_predictions:
            write_dict(losses, self.pred_outpath)

        return mean_losses
