import argparse
from utils import write_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="manually annotate images")
    parser.add_argument('--generate_template', type=bool, default=False, help='set to True if you wish to generate a template for annotating')
    parser.add_argument('--outfile', help='if generate_template is True, set to the destination of template')
    parser.add_argument('--infile', help='if generate_template is False, set to the destination of completed annotation template')
    parser.add_argument('--coco_annot_file', help='if generate template is False, set to the destination of the coco annotation file')

    args = parser.parse_args()
    generate_template = args.generate_template
    if generate_template:
        outfile = args.outfile
    else:
        infile = args.infile

