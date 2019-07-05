#%% [markdown]
# <h1> Create TensorFlow model </h1>
# 
# This notebook illustrates:
# <ol>
# <li> Creating a model using the high-level Estimator API 
# </ol>

#%%
# change these to try this notebook out
BUCKET = 'qw-gcp-a52a89d78c80969a'
PROJECT = 'qwiklabs-gcp-a52a89d78c80969a'
REGION = 'europe-west1'


#%%
import os
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION


#%%
get_ipython().run_cell_magic('bash', '', 'if ! gsutil ls | grep -q gs://${BUCKET}/; then\n  gsutil mb -l ${REGION} gs://${BUCKET}\nfi')

#%% [markdown]
# <h2> Create TensorFlow model using TensorFlow's Estimator API </h2>
# <p>
# First, write an input_fn to read the data.
# <p>
# 
# ## Lab Task 1
# Verify that the headers match your CSV output

#%%
import shutil
import numpy as np
import tensorflow as tf


#%%
# Determine CSV, label, and key columns
CSV_COLUMNS = 'weight_pounds,is_male,mother_age,plurality,gestation_weeks,key'.split(',')
LABEL_COLUMN = 'weight_pounds'
KEY_COLUMN = 'key'

# Set default values for each CSV column
DEFAULTS = [[0.0], ['null'], [0.0], ['null'], [0.0], ['nokey']]
TRAIN_STEPS = 1000

#%% [markdown]
# ## Lab Task 2
# 
# Fill out the details of the input function below

#%%
# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def read_dataset(filename_pattern, mode, batch_size = 512):
  def _input_fn():
    def decode_csv(line_of_text):
      # TODO #1: Use tf.decode_csv to parse the provided line
      # TODO #2: Make a Python dict.  The keys are the column names, the values are from the parsed data
      # TODO #3: Return a tuple of features, label where features is a Python dict and label a float
      columns = tf.decode_csv(line_of_text)
      features = dict(zip(CSV_COLUMNS,columns))
      label = features.pop('weight_pounds')
      return features, label
    
    # TODO #4: Use tf.gfile.Glob to create list of files that match pattern
    file_list = tf.gfile.Glob(filename_pattern)

    # Create dataset from file list
    dataset = (tf.data.TextLineDataset(file_list)  # Read text file
                 .map(decode_csv))  # Transform each elem by applying decode_csv fn
    
    # TODO #5: In training mode, shuffle the dataset and repeat indefinitely
    #                (Look at the API for tf.data.dataset shuffle)
    #          The mode input variable will be tf.estimator.ModeKeys.TRAIN if in training mode
    #          Tell the dataset to provide data in batches of batch_size 

    
    # This will now return batches of features, label
    return dataset
  return _input_fn


#%%
model = tf.estimator.LinearRegressor(teatcols)

#%% [markdown]
# ## Lab Task 3
# 
# Use the TensorFlow feature column API to define appropriate feature columns for your raw features that come from the CSV.
# 
# <b> Bonus: </b> Separate your columns into wide columns (categorical, discrete, etc.) and deep columns (numeric, embedding, etc.)

#%%
# Define feature columns

#%% [markdown]
# ## Lab Task 4
# 
# To predict with the TensorFlow model, we also need a serving input function (we'll use this in a later lab). We will want all the inputs from our user.
# 
# Verify and change the column names and types here as appropriate. These should match your CSV_COLUMNS

#%%
# Create serving input function to be able to serve predictions later using provided inputs
def serving_input_fn():
    feature_placeholders = {
        'is_male': tf.placeholder(tf.string, [None]),
        'mother_age': tf.placeholder(tf.float32, [None]),
        'plurality': tf.placeholder(tf.string, [None]),
        'gestation_weeks': tf.placeholder(tf.float32, [None])
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

#%% [markdown]
# ## Lab Task 5
# 
# Complete the TODOs in this code:

#%%
# Create estimator to train and evaluate
def train_and_evaluate(output_dir):
  EVAL_INTERVAL = 300
  run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL,
                                      keep_checkpoint_max = 3)
  # TODO #1: Create your estimator
  estimator = None
  train_spec = tf.estimator.TrainSpec(
                       # TODO #2: Call read_dataset passing in the training CSV file and the appropriate mode
                       input_fn = None,
                       max_steps = TRAIN_STEPS)
  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
  eval_spec = tf.estimator.EvalSpec(
                       # TODO #3: Call read_dataset passing in the evaluation CSV file and the appropriate mode
                       input_fn = None,
                       steps = None,
                       start_delay_secs = 60, # start evaluating after N seconds
                       throttle_secs = EVAL_INTERVAL,  # evaluate every N seconds
                       exporters = exporter)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

#%% [markdown]
# Finally, train!

#%%
# Run the model
shutil.rmtree('babyweight_trained', ignore_errors = True) # start fresh each time
tf.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file
train_and_evaluate('babyweight_trained')

#%% [markdown]
# The exporter directory contains the final model.
#%% [markdown]
# <h2> Monitor and experiment with training </h2>
#%% [markdown]
# To begin TensorBoard from within AI Platform Notebooks, click the + symbol in the top left corner and select the Tensorboard icon to create a new TensorBoard.
#%% [markdown]
# In TensorBoard, look at the learned embeddings. Are they getting clustered? How about the weights for the hidden layers? What if you run this longer? What happens if you change the batchsize?
#%% [markdown]
# Copyright 2017-2018 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License

