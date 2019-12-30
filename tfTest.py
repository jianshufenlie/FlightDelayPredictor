import tensorflow as tf
import pandas as pd
import shutil

print(tf.__version__)
print(pd.__version__)

# 1. read our flight data into pandas dataframes
df_train = pd.read_csv(filepath_or_buffer="2008.csv")
df_valid = pd.read_csv(filepath_or_buffer="2007.csv")

CSV_COLUMN_NAMES = list(df_train)
print(CSV_COLUMN_NAMES)

# todo: code to swap 14th and 1st column

# 2. locate label (i.e. ArrDelay - arrival delay col) in data. 14th column (0-indexed)
FEATURE_NAMES = CSV_COLUMN_NAMES[1:]
LABEL_NAME = CSV_COLUMN_NAMES[0]

# 3. create feature columns (it's an input for estimators)
feature_columns = [tf.feature_column.numeric_column(key = k) for k in FEATURE_NAMES]

# 4. define input function
def train_input_fn(df, batch_size = 128):
    #1. Convert pandas dataframe into (features,label) format for Estimator API
    # features are a dictionary with (K,V)=(feature name, tensor containing values)
    dataset = tf.data.Dataset.from_tensor_slices(tensors = (dict(df[FEATURE_NAMES]), df[LABEL_NAME]))
                                #^ problems w/ graph serialization??
    
    # If we returned now, the Dataset would iterate over the data once  
    # in a fixed order, and only produce a single element at a time.
    
    #2. Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(buffer_size = 1000).repeat(count = None).batch(batch_size = batch_size)
   
    return dataset

# 5. define the eval function exactly like input, but no repeat or shuffle
def eval_input_fn(df, batch_size = 128):
    dataset = tf.data.Dataset.from_tensor_slices(tensors = (dict(df[FEATURE_NAMES]), df[LABEL_NAME]))
    dataset = dataset.batch(batch_size = batch_size)   
    return dataset

#6. define the predict input function (w/ no value obviously)
def predict_input_fn(df, batch_size = 128):
    dataset = tf.data.Dataset.from_tensor_slices(tensors = dict(df[FEATURE_NAMES])) # no label
    dataset = dataset.batch(batch_size = batch_size)
    return dataset

#7. choose estimator
OUTDIR = "flight_trained"

config = tf.estimator.RunConfig(model_dir=OUTDIR, tf_random_seed=1, save_checkpoints_steps=100)
model = tf.estimator.DNNRegressor([10,10], feature_columns = feature_columns, config = config)

#8. Train
#9. Evaluate
#10. Predict
#11. Change Estimator type?
#12. Summarize Results

# todo: look into dataframe implementation, auto-1 hot encode categorical data, feature crosses and embeddings
# explore tf.data functionality 


# sources : https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive/