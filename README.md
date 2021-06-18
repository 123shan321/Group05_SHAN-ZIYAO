# group05 

## Histopathological-based  Metastatic Cancer Detection using  Deep learning

- [Problem Description](#problem-description)
- [Objective](#objective)
- [CAMELYON17 Dataset](#camelyon17-dataset)
- [Environment settings](#environment-settings)
- [Requirements](#requirements)
- [Usage](#usage)
- [Main Methods](#main-methods)
- [Contributing](#contributing)
- [License](#license)

## Problem Description

The diagnosis of breast cancer metastasis is done by observing the whole slide image (WSI) obtained by the biopsy. Because manually annotating tumors in WSI requires extensive microscopic evaluation and is highly time-consuming, the build of automated breast cancer metastasis detection and classification models can save time and manpower, and correct the shortcomings of time-consuming and manpower consumption.

## Objective

● Divide WSI into fixed-size tiles and label them as positive or negative 

● Utilize Convolutional Neural Network (CNN) to classify WSI as tumor or normal

● Generate a cancer probability map from WSI tiles

## CAMELYON17 Dataset

You can get the data at [CAMELYON17](https://camelyon17.grand-challenge.org/download/) challenge (GoogleDrive/Baidu).

● Contains 1000 whole-slide images (WSI) of lymph node sections 

● Training set and test set contain each 100 patients, and each patient consists of five WSI

● Training set contains 50 slides with detailed annotations of metastases

## Environment settings

All enviroments for data processing and training model are set up with anaconda. You should execute conda activate project to enable them. And this path(/custom_modules )needs to be added to PYTHONPATH environment variable or appended to sys.path.

## Requirements

ATTENTION: The first half of the notebooks, which are about WSI preprocessing and Tile Classification require at least 3.5 tera bytes space for the data set, and a fairly strong GPU (at least Geforce 1070 recommended), and a lot of time to run the processes. And you will also need to install openslide, which is needed to read the WSIs tif format.

Install using Conda (Recommended - Tested on Ubuntu 18.04.5)

Conda Install - Openslide and Python Packages

Using Conda on Ubuntu 18.04.5 (Tested):

Use the following commands to create and activate a conda environment:

```
conda create -n myenv
conda activate myenv
```

Then install the following packages via pip/conda (note to NOT use the `--user` flag for `pip install` inside a conda env!

```
conda install scipy=1.5.4
conda install tensorflow-gpu=1.14
conda install -c bioconda openslide=3.4.1
conda install -c conda-forge libiconv=1.15
conda install -c bioconda openslide-python==1.1.1
conda install python-graphviz=0.13.2
pip install openslide-python==1.1.1
pip install progress==1.5
pip install scikit-image==0.16.2
pip install scikit-learn==0.21.3
pip install pandas=0.25.3
```

## Usage

The code usage guide is shown below:

1. Move to preprocess & run `generate_tiles.py`
   -> Extract tiles from WSIs

2. Run `generate_hdf5.py`
   -> Collect all tiles into a single hdf5 file

##### NOTE: The generated files of steps 1 and 2 are in (project/data_generated). Because the file is too large and not all uploaded, please create the folder yourself when it is running.

3. Move to cnn_model & run `model.py`
   -> Train and save classifier model

##### NOTE: The generated file of steps 3（Nope model） is in (project/data_generated). Since the file is too large and has not been uploaded, you can get the trained model [here](https://drive.google.com/file/d/1D3ZgWnOlkWNJkMFwbR7I3GEpAH7uL-Rw/view?usp=sharing).

4. Move to heatmap_creation & run `heatmap.py`
   -> Generate predicted heatmap and ground truth (mask)


## Main Methods

### Data Preprocessing_Tile extraction from WSIs

With an average size of `200.000 x 100.000` pixels per WSI at the highest zoom level, it is impossible to directly train a CNN to predict the labels `negative`. Therefore, the problem has to be divided into sub tasks. Extract smaller pieces from WSIs that the WSIs are divided into `smaller pieces (tiles)` with a fixed size, e.g. `256 x 256` pixels. Each tiles is labeled with `positive` or `negative`. 

Our training data is stored as single `hdf5` file. `hdf5` stores data as key-dataset pair, similar to `dictionary` in python. Each key stands for each slide, and each dataset consists of tile for the slide. The keys are named as following:

```
"patient_#_node_#_tumor" # positive slide
"patient_#_node_#_normal" # negative slide
```

For example, `patient_020_node_4_tumor` and `patient_021_node_2_normal`.

Corresponding dataset is 4-dimensional array, shape of which is `(n, 256, 256, 3)` `n` is number of tiles, and `256, 256` is size of image by pixels, and `3` is the number of color channels. (R, G, and B)

### Learning Algorithm (CNN)_Feature embedding

With normalized and augmented tiles, we can train a CNN to predict whether a tile contains metastases or not.


### Probability map generation


##
## Contributing



## License

[MIT © Richard McRichface.](../LICENSE)
