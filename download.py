import fiftyone.zoo as foz
import argparse


def download_coco(classes, num_samples, directory):
    return foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections", "segmentations"],
        classes=classes,
        max_samples=num_samples,
        dataset_dir=directory,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="download dataset")
    parser.add_argument('dataset')
    parser.add_argument('directory')
    parser.add_argument('--classes', nargs="+")
    parser.add_argument('--numsamples', type=int)

    args = parser.parse_args()
    dataset = args.dataset
    directory = args.directory
    if dataset == "coco":
        download_coco(args.classes, args.numsamples, directory)

