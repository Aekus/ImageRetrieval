import clip
import torch
import numpy as np
from utils import load_image, write_dict
from itertools import zip_longest
from models.base import BaseModel
from tqdm import tqdm

class UpdateDirModel(BaseModel):
    def __init__(self, args):
        super(UpdateDirModel, self).__init__(args)
        self.gamma = args["gamma"]

    def predict(self, feedbacks, encoded_u_prev=None):
        if self.encoded_images is None:
            self._create_image_embeddings()

        with torch.no_grad():
            if encoded_u_prev is None:
                utterance = ' '.join(feedbacks)
                text = clip.tokenize([utterance]).to(self.device)

                best_score = 0
                best_index = 0

                encoded_text = self.model.encode_text(text).float()
                encoded_text = torch.nn.functional.normalize(encoded_text, dim=1)

                eval_vector = torch.matmul(self.encoded_images, torch.t(encoded_text))

                best_score = torch.max(eval_vector)
                best_index = torch.argmax(eval_vector)

                return self.paths[best_index], best_score.item(), encoded_text
            else:
                u_next = clip.tokenize([feedbacks[-1]]).to(self.device)
                encoded_u_next = self.model.encode_text(u_next).float()
                encoded_u_next = torch.nn.functional.normalize(encoded_u_next, dim=1)

                prev_eval_vector = torch.matmul(self.encoded_images, torch.t(encoded_u_prev))

                prev_pred_image_id = torch.argmax(prev_eval_vector)

                epsilon = 1e-5
                while True:
                    encoded_u = encoded_u_prev + epsilon * encoded_u_next
                    encoded_u = torch.nn.functional.normalize(encoded_u, dim=1)

                    eval_vector = torch.matmul(self.encoded_images, torch.t(encoded_u))

                    if torch.argmax(eval_vector) != prev_pred_image_id:
                        break

                    epsilon *= self.gamma

                best_score = torch.max(eval_vector)
                best_index = torch.argmax(eval_vector)

                return self.paths[best_index], best_score.item(), encoded_u

    def eval(self, annotations, write_predictions=True):
        if self.encoded_images is None:
            self._create_image_embeddings()

        predictions = {"predictions": []}

        self.logger.info("making predictions")
        for annot in tqdm(annotations["annotations"]):
            path = annot["source"]
            feedbacks = annot["feedbacks"]
            predictions["predictions"].append({"id": annot["id"],
                                               "source": path,
                                               "losses": [],
                                               "predicted outputs": [],
                                               "clip scores": []})
            prediction = predictions["predictions"][-1]

            encoded_u = None
            for i in range(1, len(feedbacks) + 1):
                pred_path, score, encoded_u = self.predict(feedbacks[:i], encoded_u_prev=encoded_u)

                prediction["predicted outputs"].append(pred_path)
                prediction["clip scores"].append(score)

                loss = self.loss(load_image(path), load_image(pred_path))
                prediction["losses"].append(loss)

                if pred_path == path:
                    prediction["predicted outputs"].extend([pred_path] * (len(feedbacks) - i))
                    prediction["clip scores"].extend([score] * (len(feedbacks) - i))
                    prediction["losses"].extend([loss] * (len(feedbacks) - i))

                    break

        self.logger.info("finished making predictions")

        losses = [prediction["losses"] for prediction in predictions["predictions"]]

        losses_t = [list(filter(None, i)) for i in zip_longest(*losses)]

        mean_losses = []
        for i in range(len(losses_t)):
            mean_losses.append(np.mean(losses_t[i]))

        if write_predictions:
            write_dict(self.pred_outpath, predictions)

        return mean_losses, predictions

