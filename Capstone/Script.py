from Capstone import Prepare, Preprocess, Train, Predict
import logging

logging.basicConfig(filename='DL_Log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

file = "D:\MLE Capstone Project\Data\DL_data_prepared_multigroup.csv"""
train, valid, test, weights, bias = Prepare.prepare_data(file)
inputs, layers, train_ds, valid_ds, test_ds = Preprocess.preprocess_data(train, valid, test)
model = Train.make_model(inputs, layers, output_bias=bias)
model = Train.fit_model(model, train_ds, valid_ds, weights)
Predict.evaluate_model(model, test_ds)
y = test['Readmit']
yhat = Predict.predict_from_model(model, test_ds)