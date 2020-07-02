# M5 Forecasting - Accuracy
Guidelines: https://mofc.unic.ac.cy/m5-competition/

- Estimate the point forecasts of the unit sales of various products sold in the USA by Walmart
- In this competition, the fifth iteration, you will use hierarchical sales data from Walmart, the world’s largest company by revenue, to **forecast daily sales for the next 28 days**. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. Together, this robust dataset can be used to improve forecasting accuracy.
- The historical data range from  2011-01-29  to  2016-06-19

# Files
- `calendar.csv` - Contains information about the dates on which the products are sold.
- `sales_train_validation.csv` - Contains the historical daily unit sales data per product and store [d_1 - d_1913]
- `sample_submission.csv` - The correct format for submissions. Reference the Evaluation tab for more info.
- `sell_prices.csv` - Contains information about the price of the products sold per store and date.
- `sales_train_evaluation.csv` - Available once month before competition deadline. Will include sales [d_1 - d_1941]

# Data details
- 42,840 time series
- 3,049 products
- 3 product categories (Hobbies, Foods, and Household)
- 7 product departments
- 10 stores
- 3 states (CA, TX, WI)

# LGBM
- Under the hood: https://www.kaggle.com/cdeotte/200-magical-models-santander-0-920/

# Neural Nets
- A lot of things which I'm trying to do: https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43795
- https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88929
- http://papers.nips.cc/paper/7412-regularization-learning-networks-deep-learning-for-tabular-datasets.pdf
- Tabular pytorch: https://github.com/yashu-seth/pytorch-tabular/blob/master/pytorch_tabular.py
- Winning NN santander: https://www.kaggle.com/fl2ooo/nn-wo-pseudo-1-fold-seed
- Fast loader: https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
- https://arxiv.org/pdf/1803.09820.pdf



# Read 
- Deal with zeros: https://seananderson.ca/2014/05/18/gamma-hurdle/
- Deal with zeros: https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/82614


# Ideas
- Kalman filters (https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43727)
- Direct approach or not: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/139461

# Custom loss
- backprop custom loss: https://www.kaggle.com/mithrillion/know-your-objective 
- Discussions on custom loss: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/71328
- Fourier feature engineering

# Hierarchy
- https://robjhyndman.com/publications/nnmint/ (IP)


# FE:
- Rolling target encoding mean/std for department and other ids
- Weighted running average


# Literature review
P: Pristine, I haven't read it yet.
S: Started.
C: Completed.

Guidelines: https://mofc.unic.ac.cy/m5-competition/

- P: https://www.lancaster.ac.uk/pg/waller/pdfs/Intermittent_Demand_Forecasting.pdf (A good survey of many methods)
- P: https://medium.com/analytics-vidhya/croston-forecast-model-for-intermittent-demand-360287a17f5f (croston for intermittent)
- P: https://otexts.com/fpp2/hierarchical.html
- P: https://www.sciencedirect.com/science/article/pii/S0167947311000971
- P: http://webdoc.sub.gwdg.de/ebook/serien/e/monash_univ/wp9-07.pdf
- P: https://forecasters.org/wp-content/uploads/gravity_forms/7-2a51b93047891f1ec3608bdbd77ca58d/2014/07/Athanasopoulos_George_ISF2014.pdf
- P: https://medium.com/opex-analytics/hierarchical-time-series-101-734a3da15426
- P: https://arxiv.org/pdf/1912.00370v1.pdf
- P: https://stats.stackexchange.com/questions/24339/how-to-detect-intermittent-time-series
