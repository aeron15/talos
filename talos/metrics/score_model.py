from sklearn.metrics import roc_auc_score
from keras.callbacks import *
from .performance import Performance
import pdb
import pandas
import numpy

def get_score(self):

    pdb.set_trace()

    y_pred = self.model.predict(self.x_val)
    # y_pred = self.kerao_model.predict_classes(self.x_val)
    return Performance(y_pred, self.y_val, self.shape, self.y_max).result
