import json
from PIL import Image

"""
JSON utils
"""


def write_dict(json_file, data):
    with open(json_file, 'w') as outfile:
        json.dump(data, outfile)


def load_dict(json_file):
    with open(json_file, 'r') as infile:
        data = json.load(infile)
    return data


"""
Bounding Box utils
"""


def get_area(x1, y1, x2, y2):
    return abs(x1 - x2) * abs(y1 - y2)


"""
Image utils
"""


def load_image(path):
  return Image.open(path)