## Getting started

Create and activate the conda environment

```
conda env create -f environment.yml && conda activate chiron
```

Open up Jupyter notebook

```
jupyter-notebook --notebook-dir notebooks
```

Run the notebooks [`prepare-cheng-et-al-data.ipynb`](https://github.com/mattmolinare/chiron/blob/main/notebooks/prepare-cheng-et-al-data.ipynb) and [`prepare-kaggle-data.ipynb`](https://github.com/mattmolinare/chiron/blob/main/notebooks/prepare-kaggle-data.ipynb) to generate TFRecord files from the raw data. The notebook [`combined-data.ipynb`](https://github.com/mattmolinare/chiron/blob/main/notebooks/combine-data.ipynb) can be used to combine the two datasets into a single dataset containing both positive and negative examples.

## Downloading the data

The dataset published Cheng et al. contains images of three type of brain tumors: meningioma, glioma, and pituitary[^cheng-et-al]. To download it, run the command

```
(cd data/cheng-et-al && sh download.sh)
```

[This Kaggle dataset](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset) contains images of brains without tumors. Download it via

```
(cd data/kaggle && sh download.sh)
```

[^cheng-et-al]: Cheng, Jun, et al. "Enhanced performance of brain tumor classification via tumor region augmentation and partition." PloS one 10.10 (2015): e0140381.
