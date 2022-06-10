import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import class_weight
import logging

def prepare_data(csv_file):
    data = pd.read_csv(csv_file)
    logging.info("The data has been loaded.")
    
    data['Readmit'] = np.where(data['Readmit'] == 2, 1, data['Readmit'])
    data['Log_TOTCHG'] = np.log(data.pop('TOTCHG') + 1)
    data['Log_N_DISC_U'] = np.log(data.pop('N_DISC_U'))
    data['Log_N_HOSP_U'] = np.log(data.pop('N_HOSP_U'))
    data['Log_TOTAL_DISC'] = np.log(data.pop('TOTAL_DISC'))
    data['Log_LOS'] = np.log(data.pop('LOS') + 1)
    data['Log_I10_NPR'] = np.log(data.pop('I10_NPR') + 1)
    encoder = OrdinalEncoder()
    feature = data.pop('Primary_dx').to_numpy().reshape(-1, 1)
    Primary_dx = encoder.fit_transform(feature)
    data['Primary_dx'] = Primary_dx
    Secondary_dx = data.pop('Secondary_dx')
    Procedure = data.pop('Procedure')
    MBD_dx = data.pop('MBD_dx')
    data = data.astype('float64')
    data['Secondary_dx'] = Secondary_dx
    data['Procedure'] = Procedure
    data['MBD_dx'] = MBD_dx
    logging.info("The data has been prepared.")

    train, valid, test = np.split(data.sample(frac=1),[int(0.8*len(data)),int(0.9*len(data))])
    logging.info("The data has been split into training, validation, and test sets.")

    w0, w1 = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(train['Readmit']),y=train['Readmit'])
    class_weights = {0:w0, 1:w1}
    class_weights
    logging.info("The class weights for training have been computed.")

    neg, pos = np.bincount(train['Readmit'])
    initial_bias = np.log([pos/neg])
    logging.info("The initial bias for training has been computed.")

    return train, valid, test, class_weights, initial_bias