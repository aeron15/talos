from sklearn.metrics import roc_auc_score
from keras.callbacks import *
# from .metrics.performance import Performance
import pdb
import pandas
import numpy

def get_score(self):

    y_pred = self.keras_model.predict(self.x_val)

    # validation_df = pandas.DataFrame({'actual': self.y_val, 'prediction': y_pred}).dropna()
    validation_df = pandas.DataFrame({'actual': self.y_val.flatten(), 'prediction': y_pred.flatten()}).dropna()

    try:
        roc_val = roc_auc_score(validation_df.actual, validation_df.prediction)
    except:
        roc_val = numpy.nan

    auc_df  = pandas.DataFrame([])
    
    n_tasks = self.y_val.shape[1]
    
    for col_idx in np.arange(n_tasks):
    
        tmp_df = pandas.DataFrame({'actual': self.y_val[:, col_idx],
                                   'prediction': y_pred[:, col_idx]}).dropna()
        
        try:
            roc_auc  = roc_auc_score(tmp_df.actual, tmp_df.prediction)
        except:
            roc_auc = numpy.nan
            pass

        auc_df = auc_df.append(pandas.DataFrame({'task': [col_idx], 'auc_per_task': [roc_auc]}))
    
    print('\n ** AUC per task ** \n')

    print(auc_df.head(n = n_tasks))

    print('\n ** Average Validation AUC: %s **' % str(auc_df['auc_per_task'].mean()))
    
    print('\n  *** ROC AUC Validation: %s ***' % str(roc_val))


    return auc_df['auc_per_task'].mean() 

    
    # ypred = self.keras_model.predict_classes(self.x_val)
    # return Performance(y_pred, self.y_val, self.shape, self.y_max).result
