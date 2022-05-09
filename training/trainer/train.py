import subprocess
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

# vertex AI automatically populates the environment variable PROJECT_ID and mounts cloud storage under /gcs
PROJECT_ID='bt-pp-dsc-1th4'  # TODO: enter your project_id here
model_output_path = '/gcs/' + PROJECT_ID + '-model-bucket/lightgbm_booster.txt'  

# The Auto MPG dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/). 
# We use a copy stored in a public cloud storage bucket.
dataset_path = 'https://storage.googleapis.com/io-vertex-codelab/auto-mpg.csv'
dataset = pd.read_csv(dataset_path, na_values='?')
dataset = dataset.drop(columns='car name').dropna()  # drop car name column and any rows containing NA values
dataset = dataset.rename(columns=lambda colname: colname.replace(' ', '_'))  # replace spaces with underscores in column names to avoid problems with LightGBM

# Split the data into train and test
train = dataset.sample(frac=0.8, random_state=0)
test = dataset.drop(train.index)

# Split features from labels
train_labels = train.pop('mpg')
test_labels = test.pop('mpg')

# Convert train and test data to lightGBM format
dtrain = lgb.Dataset(data=train, label=train_labels, categorical_feature=['origin'])
dtest = lgb.Dataset(data=test, label=test_labels, reference=dtrain)

# Now we train the model. Unlike real applications, we don't do hyperparameter optimization.
training_parameters = {'objective': 'rmse', 'num_leaves': 16, 'num_trees': 50, 'learning_rate': 0.1, 'seed': 4711, 'min_data_in_leaf': 10, 'early_stopping_round': 5}
model = lgb.train(params=training_parameters, train_set=dtrain, valid_sets=[dtest])

# Export model to cloud storage bucket
model.save_model(filename=model_output_path)