from models.base import BaseModel
import clip
import torch


class FullConcatModel(BaseModel):

    def predict(self, feedbacks):
        utterance = ' '.join(feedbacks)
        text = clip.tokenize([utterance]).to(self.device)

        best_score = 0
        best_index = 0
        for i in range(0, len(self.paths), self.batchsize):
            batch = self.paths[i:i + self.batchsize]
            images = self.image_list_to_tensor(batch)

            logits = self.model(images, text)[1]
            if torch.max(logits) > best_score:
                best_score = torch.max(logits)
                best_index = i + torch.argmax(logits)

        return self.paths[best_index], best_score
