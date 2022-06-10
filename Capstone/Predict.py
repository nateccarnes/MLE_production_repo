import pickle
from tensorflow.keras import Model
import logging

def predict_from_pickle(pickle, data, batch=512):
    model = pickle.load(open(pickle, 'rb'))
    predictions = model.predict(data, batch_size=batch, verbose=0)
    logging.info("Predictions have been made.")
    return predictions

def predict_from_model(model, data, batch=512):
    predictions = model.predict(data, batch_size=batch, verbose=0)
    logging.info("Predictions have been made.")
    return predictions

def evaluate_model(model, data, batch=512):
    model_results = model.evaluate(data, batch_size=batch, verbose=0)
    logging.info("Model evaluation metrics:")
    for name, value in zip(model.metrics_names, model_results):
        logging.info("{}: {}".format(name, value))