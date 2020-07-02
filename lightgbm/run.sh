# python make_data.py -n 3.2
python train.py -use_custom_loss yes -submit_kaggle False -random_fold True -recursive False -outdir /home/timetraveller/Work/m5data/tests -num_folds 3 -num_boost_round 1200 -early_stopping_rounds 30

# python train.py -use_custom_loss no -submit_kaggle True -random_fold True -recursive True

# python train_1.py -recursive False -num_boost_round 1500 -calc_wrmsse True -use_custom_loss False -early_stopping_rounds 110 -num_folds 3
# python train.py -recursive False -num_boost_round 2000 -calc_wrmsse True -use_custom_loss False -early_stopping_rounds 110 -num_folds 3

# python train.py -use_custom_loss True -random_fold True -early_stopping_rounds 50 -submit_kaggle True -calc_wrmsse True
# python train.py -use_custom_loss False -random_fold True -early_stopping_rounds 50 -submit_kaggle True -calc_wrmsse True

# python train_2.py -use_custom_loss True -early_stopping_rounds 50 -submit_kaggle False -num_boost_round 20