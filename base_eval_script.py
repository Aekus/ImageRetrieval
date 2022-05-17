import torch
from utils import listdir_jpg_paths, load_dict
from models.full_concat import FullConcatModel

args = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "paths": listdir_jpg_paths("CUB_200_2011/CUB_200_2011/images/validation/data"),
    "batchsize": 64,
    "pred_outpath": "cub_fcm_predictions.json"
}

fcm = FullConcatModel(args)
annotations = load_dict("cub_annotations.json")

mean_losses, predictions = fcm.eval(annotations, write_predictions=True)
print(mean_losses)