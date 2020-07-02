# Kaggle - Timeseries forecasting using DL (M5 Competition)
[Kaggle: M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/)
This is the code for timeseries forecasting using deep learning. I failed to win any medal but logging this code here for future use. (Making it public in case it might help someone)

# Notes
I tried 3 models in the following order:
## LightGBM: 
See `lightgbm` for this. 
Nothing special, mostly taken from public. Doesn't work very well, I was mostly experimenting with DL models and not this.

## Attention is all you need
See `transformer` for this.
- Implemented (with help of public repos) this paper for M5.
- Minor changes:
    1. Larger encoder length (4x decoder).
    2. Downsampling encoded output to decoded output using 1D convs (weighted average).
    3. Encoding using convolutions (over past sales, yearly upto 3 years, half yearly upto 3 years and quater yearly upto 3 years).
    4. Removed positional encoding and simply used time series' dense features with linear layers for encoding.
    5. Tried and failed with `siren`, doesn't work.
    6. `LeakyReLU` is great.
- Just see the `notebook.ipynb` for quick overview.    

## Convolutional net
See `convnet` for this.
- 1D weekly, bi-weekly, monthly, last week and bi-week convolutions for each item.
- Tried and failed with item entity encoding.
- Concatenation with dense features to get outputs.
- `2**11` batch size works great + GPUs work very well (fast) for large batch sizes. Problems with batchnorms but fixed later.
- Just see the `notebook.ipynb` for quick overview.    


## Behelit
![](https://i.imgur.com/GlnBF4D.png)