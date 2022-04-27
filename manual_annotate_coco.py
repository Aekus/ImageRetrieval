import argparse
import jsonlines
from utils import load_dict, write_dict


def make_template(outfile, annot_file):
    annotations = load_dict(annot_file)
    outlist = []
    for annotation in annotations["annotations"]:
        outlist.append({"path": annotation["source"], "feedbacks": []})

    with jsonlines.open(outfile, 'w') as writer:
        writer.write_all(outlist)


def update_annots(infile, annot_file):
    annotations = load_dict(annot_file)
    with jsonlines.open(infile) as reader:
        for i, obj in enumerate(reader):
            annotations["annotations"][i]["feedbacks"].extend(obj["feedbacks"])

    write_dict(annot_file, annotations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="manually annotate images")
    parser.add_argument('--generate_template', type=bool, default=False,
                        help='set to True if you wish to generate a template for annotating')
    parser.add_argument('--outfile', help='if generate_template is True, set to the destination of template')
    parser.add_argument('--infile',
                        help='if generate_template is False, set to the destination of completed annotation template')
    parser.add_argument('--coco_annot_file', help='set to the destination of the coco annotation file')

    args = parser.parse_args()
    generate_template = args.generate_template
    annot_file = args.coco_annot_file
    if generate_template:
        outfile = args.outfile
        make_template(outfile, annot_file)
    else:
        infile = args.infile
        update_annots(infile, annot_file)
