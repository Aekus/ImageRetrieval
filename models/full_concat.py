from models.base import BaseModel
import clip
import torch


class FullConcatModel(BaseModel):

    def predict(self, feedbacks):
        if self.encoded_images is None:
            self._create_image_embeddings()

        with torch.no_grad():
            utterance = ' '.join(feedbacks)
            text = clip.tokenize([utterance]).to(self.device)

            best_score = 0
            best_index = 0

            encoded_text = self.model.encode_text(text)
            eval_vector = torch.matmul(self.encoded_images, torch.t(encoded_text))

            best_score = torch.max(eval_vector)
            best_index = torch.argmax(eval_vector)

            return self.paths[best_index], best_score.item()
