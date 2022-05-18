import argparse
import numpy as np
from download import download_cub
from utils import get_area
from utils import write_dict


def make_class_annotations(dataset, numsamples, max_depth=3):
    annot = {'annotations': []}
    attribute_map = dataset.attribute_map
    label_map = dataset.label_map
    idx = np.random.choice(np.arange(len(dataset)), size=numsamples, replace=False)
    for i in idx:
        sample = dataset[i]
        np.random.shuffle(sample["attributes"])
        feedbacks = []
        label_string = dataset.pre_process_labels_and_annotations(label_map[sample["label"]])
        feedbacks.append(label_string)
        for attribute_id in sample["attributes"][:max_depth]:
            attribute_string = dataset.pre_process_labels_and_annotations(attribute_map[attribute_id])
            feedbacks.append(attribute_string)

        annot['annotations'].append({'id': int(sample["id"]), 'source': sample["path"], 'feedbacks': feedbacks})

    return annot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="annotate images")
    parser.add_argument('directory')
    parser.add_argument('numsamples', type=int)
    parser.add_argument('outfile')
    parser.add_argument('feedback_depth', type=int)

    args = parser.parse_args()
    directory = args.directory
    numsamples = args.numsamples
    feedback_depth = args.feedback_depth
    dataset = download_cub(directory)

    annotations = make_class_annotations(dataset, numsamples=numsamples, max_depth=feedback_depth)
    write_dict(args.outfile, annotations)