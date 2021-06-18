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

Our training data is stored as single `hdf5` file if anything fails. `hdf5` stores data as key-dataset pair, similar to `dictionary` in python. Each key stands for each slide, and each dataset consists of tile for the slide. The keys are named as following:

```
"patient_#_node_#_tumor" # positive slide
"patient_#_node_#_normal" # negative slide
```

For example, `patient_020_node_4_tumor` and `patient_021_node_2_normal`.

Corresponding dataset is 4-dimensional array, shape of which is `(n, 256, 256, 3)` `n` is number of tiles, and `256, 256` is size of image by pixels, and `3` is the number of color channels. (R, G, and B)

### Learning Algorithm (CNN)_Feature embedding

Define a class`TissueDataset` which handles access to our generated HDF5 file. And load data set split into training and validation data. By setting `train_data = TissueDataset(path=HDF5_FILE, percentage=0.5, first_part=True)`, we say that `training_data` consists of the first 50% of tiles of every WSI. 

```
train_data = TissueDataset(path=HDF5_FILE,  percentage=0.5, first_part=True)
val_data = TissueDataset(path=HDF5_FILE, percentage=0.5, first_part=False)
x, y = train_data.get_batch(num_neg=3, num_pos=3)
```

Use the method `get_batch` returns a specified number of positive and negative slides. The slides are randomly shuffeled, so the first part of the batch does not only contain positive slides, and the last part of the batch only negtive slides.

```
def get_batch(self, num_neg=10, num_pos=10, data_augm=False):
    x_p, y_p = self.__get_random_positive_tiles(num_pos)
    x_n, y_n = self.__get_random_negative_tiles(num_neg)
    x = np.concatenate((x_p, x_n), axis=0)
    y = np.concatenate((y_p, y_n), axis=0)
    if data_augm:
       ### some data augmentation mirroring / rotation
       if np.random.randint(0,2): x = np.flip(x, axis=1)
       if np.random.randint(0,2): x = np.flip(x, axis=2)
       x = np.rot90(m=x, k=np.random.randint(0,4), axes=(1,2))
        ### randomly arrange in order
    p = np.random.permutation(len(y))
    return x[p], y[p]
```

The `generator` method does the same as `get_batch`, but implements a python generator, which is very useful when training and evaluating a `tensorflow.keras` model.

```   
def generator(self, num_neg=10, num_pos=10, data_augm=False, mean=[0.,0.,0.], std=[1.,1.,1.]):
    while True:
        x, y = self.get_batch(num_neg, num_pos, data_augm)
        for i in [0,1,2]:
            x[:,:,:,i] = (x[:,:,:,i] - mean[i]) / std[i]
        yield x, y
```

The method argument `data_augm=True` randomly rotates the batch zero to three times by 90 degrees and randomly flips is horizonally and / or vertically. Make the slides have been successfully stored in the HDF5 file, and enable them to be successfully loaded with the label.

#### Defining the Convolution Neural Network

Define a model for the training. Use classes / methods from `tensorflow.keras`. Also compile model with an optimizer, a loss and a metric (e.g. accuracy). As we only have two classes, we can use the binary crossentropy as loss function.

``` 
base_model = keras.applications.InceptionResNetV2(
                                 include_top=False, 
                                 weights='imagenet', 
                                 input_shape=(256,256,3), 
                                 )
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
predictions = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

The next step is to train and evaluate the model. Keep track the metrics loss and accuracy of training and validation data, and plot these data later.

Make checkpoints every epoch. And use the `fit_generator` method of your model and pass `generator` method as parameter.

The following parameters showed to work well:

1. Each epoch consists of 50 training batches and 25 validations batches
2. Each batch contains 10 negatives and 10 positives
3. Train for at least 25 epochs

```
batch_size_neg=10
batch_size_pos=10
batches_per_train_epoch = 50
batches_per_val_epoch = 25
epochs = 25
```

Finally save the trained model.

##### NOTE: The generated file of Nope model is in (project/data_generated). Since the file is too large and has not been uploaded, you can get the trained model [here](https://drive.google.com/file/d/1D3ZgWnOlkWNJkMFwbR7I3GEpAH7uL-Rw/view?usp=sharing).

```
model.save(MODEL_FINAL)
```

### Probability map generation


##
## Contributing



## License

[MIT © Richard McRichface.](../LICENSE)
