import argparse
from download import download_coco
from utils import get_area
from utils import write_dict


def make_class_annotations(dataset, max_depth=3):
    annot = {'annotations': []}
    for i, sample in enumerate(dataset):
        objects = sample.detections.detections
        label_area_map = {}
        for obj in objects:
            bb = obj.bounding_box
            area = get_area(bb[0], bb[1], bb[2], bb[3])
            if obj.label not in label_area_map:
                label_area_map[obj.label] = area
            else:
                label_area_map[obj.label] = max(label_area_map[obj.label], area)

        sorted_classes = sorted(label_area_map.keys(), key=lambda k: label_area_map[k])
        annot['annotations'].append({'id': sample.id, 'source': sample.filepath, 'feedbacks': sorted_classes[:max_depth]})

    return annot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="annotate images")
    parser.add_argument('directory')
    parser.add_argument('classes', nargs="+")
    parser.add_argument('numsamples', type=int)
    parser.add_argument('outfile')

    args = parser.parse_args()
    directory = args.directory
    dataset = download_coco(args.classes, args.numsamples, directory)

    annotations = make_class_annotations(dataset)
    write_dict('outfile', annotations)
