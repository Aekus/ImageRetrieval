import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
from matplotlib import pyplot as plt

def show_image(image):
  plt.figure()
  imgplot = plt.imshow(image)
  plt.show()
