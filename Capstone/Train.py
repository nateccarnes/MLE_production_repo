from tensorflow import keras
import pickle
import pandas as pd
import logging

my_metrics = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall')
]

def make_model(inputs, encodings, metrics=my_metrics, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    initializer = keras.initializers.HeUniform()
    all_features = keras.layers.concatenate(encodings)
    Dense1 = keras.layers.Dense(256, activation="relu", kernel_initializer=initializer)(all_features)
    Norm1 = keras.layers.BatchNormalization()(Dense1)
    Dropout1 = keras.layers.Dropout(0.5)(Norm1)
    Dense2 = keras.layers.Dense(128, activation="relu", kernel_initializer=initializer)(Dropout1)
    Norm2 = keras.layers.BatchNormalization()(Dense2)
    Dropout2 = keras.layers.Dropout(0.5)(Norm2)
    Dense3 = keras.layers.Dense(64, activation="relu", kernel_initializer=initializer)(Dropout2)
    Norm3 = keras.layers.BatchNormalization()(Dense3)
    Dropout3 = keras.layers.Dropout(0.5)(Norm3)
    Dense4 = keras.layers.Dense(32, activation="relu", kernel_initializer=initializer)(Dropout3)
    Norm4 = keras.layers.BatchNormalization()(Dense4)
    Dropout4 = keras.layers.Dropout(0.5)(Norm4)
    output = keras.layers.Dense(1, activation="sigmoid", bias_initializer=output_bias)(Dropout4)
    model = keras.Model(inputs, output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=metrics)
    logging.info("The model has been compiled.")
    return model

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=0,
    patience=10,
    mode='max',
    restore_best_weights=True)

def fit_model(model, train_data, valid_data, weights, batch=2048, epoch=30):
    history = model.fit(x=train_data, validation_data=valid_data, verbose=0,
    class_weight=weights, batch_size=batch, epochs=epoch)
    logging.info("The model has been trained.")
    pickle.dump(model, open('DL_model.pkl', 'wb'))
    logging.info("The model has been saved.")
    hist_df = pd.DataFrame(history.history)
    with open('Training_History.csv', mode='w') as f:
        hist_df.to_csv(f)
    logging.info("The training history has been saved.")
    return model