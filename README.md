# group05 

## Histopathological-based  Metastatic Cancer Detection using  Deep learning

- [Problem Description](#problem-description)
- [Objective](#objective)
- [CAMELYON17 Dataset](#camelyon17-dataset)
- [Environment settings](#environment-settings)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Problem Description

The diagnosis of breast cancer metastasis is done by observing the whole slide image (WSI) obtained by the biopsy. Because manually annotating tumors in WSI requires extensive microscopic evaluation and is highly time-consuming, the build of automated breast cancer metastasis detection and classification models can save time and manpower, and correct the shortcomings of time-consuming and manpower consumption.

## Objective



## CAMELYON17 Dataset

You can get the data at [CAMELYON17](https://camelyon17.grand-challenge.org/download/) challenge (GoogleDrive/Baidu).


## Environment settings

All enviroments for data processing and training model are set up with anaconda. You should execute conda activate project to enable them. And this path(/custom_modules )needs to be added to PYTHONPATH environment variable or appended to sys.path.



## Install

This module depends upon a knowledge of [Markdown]().

```
dfdsfdsfd
```

### Any optional sections

## Usage

1. Move to preprocess & run generate_tiles.py
   -> extract tiles from WSIs

2. run generate_hdf5.py
   -> collect all tiles into a single hdf5 file

3. move to cnn_model & run model.py
   -> train and save classifier model

4. move to heatmap_creation & run heatmap.py
   -> generate predicted heatmap and ground truth (mask)


## Contributing

See [the contributing file](CONTRIBUTING.md)!

PRs accepted.

Small note: If editing the Readme, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme) specification.

### Any optional sections

## License

[MIT © Richard McRichface.](../LICENSE)
