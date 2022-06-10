import tensorflow as tf
from tensorflow.keras import layers
import logging

def dataframe_to_dataset(dataframe, shuffle=True, batch_size=32):
  df = dataframe.copy()
  labels = df.pop('Readmit')
  df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

def get_normalization_layer(name, dataset):
  normalizer = layers.Normalization(axis=None)
  feature_ds = dataset.map(lambda x, y: x[name])
  normalizer.adapt(feature_ds)
  return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)
  feature_ds = dataset.map(lambda x, y: x[name])
  index.adapt(feature_ds)
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
  return lambda feature: encoder(index(feature))

def get_vectorization_layer(name, dataset, output_mode='multi_hot'):
  vectorizer = layers.TextVectorization(output_mode=output_mode)
  feature_ds = dataset.map(lambda x, y: x[name])
  vectorizer.adapt(feature_ds)
  return vectorizer

def preprocess_data(train_data, valid_data, test_data, batch_size=512):
    train_ds = dataframe_to_dataset(train_data, batch_size=batch_size)
    valid_ds = dataframe_to_dataset(valid_data, batch_size=batch_size)
    test_ds = dataframe_to_dataset(test_data, batch_size=batch_size)
    logging.info("The pd.dataframe objects have been converted to tf.dataset objects.")
    
    all_inputs = []
    encoded_features = []
    continuous_features = ['APRDRG_Risk_Mortality','APRDRG_Severity','ZIPINC_QRTL','AGE','DMONTH','I10_NDX',
                       'Log_TOTCHG','Log_N_DISC_U','Log_N_HOSP_U','Log_TOTAL_DISC','Log_LOS','Log_I10_NPR']
    categorical_features = ['APRDRG','HOSP_BEDSIZE','H_CONTRL','HOSP_URCAT4','HOSP_UR_TEACH','DIED','DISPUNIFORM','ELECTIVE',
                        'FEMALE','HCUP_ED','PAY1','PL_NCHS','REHABTRANSFER','RESIDENT','SAMEDAYEVENT','Primary_dx']
    text_features = ['Secondary_dx','Procedure','MBD_dx']
    
    for header in continuous_features:
        cont_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(name=header, dataset=train_ds)
        encoded_cont_col = normalization_layer(cont_col)
        all_inputs.append(cont_col)
        encoded_features.append(encoded_cont_col)
    logging.info("The continuous features have been preprocessed.")

    for header in categorical_features:
        cat_col = tf.keras.Input(shape=(1,), name=header, dtype='float64')
        encoding_layer = get_category_encoding_layer(name=header, dataset=train_ds, dtype='float64')
        encoded_cat_col = encoding_layer(cat_col)
        all_inputs.append(cat_col)
        encoded_features.append(encoded_cat_col)
    logging.info("The categorical features have been preprocessed.")

    for header in text_features:
        text_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        vectorization_layer = get_vectorization_layer(name=header, dataset=train_ds)
        encoded_text_col = vectorization_layer(text_col)
        all_inputs.append(text_col)
        encoded_features.append(encoded_text_col)
    logging.info("The text features have been preprocessed.")

    return all_inputs, encoded_features, train_ds, valid_ds, test_ds