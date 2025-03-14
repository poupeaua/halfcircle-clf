# Half-Circle Classifier

## Introduction

The object of this project is to train a classifier to detect if an image has a 
half-circle in it. It is quite straightforward but we need a clear and concise
repository to document the project and make it reproducible.

The project facilitate also the productionalization of the model within a Docker image (#TODO)

## Data

Since the object within the image we want to classify is easy (a half-circle), 
we can generate synthetic data automatically. This makes the tasks really accessible
to train a robust classifier. Plus only using grayscale image makes the tasks
even simpler by using only images with one input channel.

The different data used are the following:
- Basic Half circle: images generated with [generate.py](src/data/generate.py)
- Crops from .pdf file: images generated with [generate.py](src/data/generate.py).
This is particularly useful to generate data from PDF files that look like real-life scanned documents.
Once trained, the model in production will predict on crops of PDF files so
training on crops from PDF files samples is really useful. Moreover, we can also
draw half-circles on top of the crops images to not only augment the label 0 data but
also the label 1 data.
- [Quick Draw dataset](https://github.com/googlecreativelab/quickdraw-dataset?tab=readme-ov-file#get-the-data): it contains a large collection of hand-drawn grayscale 
images.
- [Picsum](https://picsum.photos/): this is a website that generates random images.
We can easily download thousands of grayscale images of any shape we want.
- Noise images: generated with [generate.py](src/data/generate.py) to make the model more robust.
- Pure filled images: generated with [generate.py](src/data/generate.py) to make the model more robust.

### Data Augmentation

Using the pytorch v2 of [torchvision.transforms](https://pytorch.org/vision/main/transforms.html)
is it really easy to augment image data. Please check the file [transforms.py](src/data/transforms.py)
to see the applied augmentations.

Plus this makes it simpler to replicate the same transformations on the test data.
This way, we make sure the inference is done on the same type of input data as in training.

## Tools

This project uses [uv](https://docs.astral.sh/uv/) as a project dependency manager.
I was used to [poetry](https://python-poetry.org/docs/) but I wanted to try it out so 
here it is in a new project.