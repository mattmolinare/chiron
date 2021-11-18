## Getting started

Create the conda environment

```
conda env create -f environment.yml
conda activate chiron
```

## Downloading the data

Download the brain tumor public dataset[^j-cheng]

```
(cd data/brain-tumor-public-dataset && ./download.sh mat)
```

[^j-cheng]: Cheng, Jun, et al. "Enhanced performance of brain tumor classification via tumor region augmentation and partition." PloS one 10.10 (2015): e0140381.
