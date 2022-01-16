# ResNet-Visualization

This program depends on "imagen" dataset -- a subset of ImageNet containing 1000 images. Imagen can be downloaded from https://github.com/ajschumacher/imagen/tree/master/imagen.

**resn1.py** is the main program. It visualizes the output of layer 1-4 of ResNet. For each output I only visualize the first channel.

**imagenet_class_index.json** is used to map the output class numbers (0 ~ 999) to their corresponding labels.

**word.txt** maps the WordNet ID to labels, although this file is not used by resn1.py.
