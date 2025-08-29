import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import logging
from dvclive import Live
import yaml

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model_evaluation.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_model(file_path:str):
    """Load the trained model from a file."""
    try:
        with open(file_path,'rb') as f:
            model=pickle.load(f)

        logger.debug('Model loaded from %s',file_path)
        return model
    except FileNotFoundError:
        logger.error('file not found: %s',file_path)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loading the model: %s',e)
        raise

def load_data(file_path:str):
    """Load data from a CSV file."""
    try:
        df=pd.read_csv(file_path)
        logger.debug('data loaded from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse the csv file: %s',e)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loading the data: %s',e)
        raise

def evaluate_model(model,x_test:np.ndarray,y_test:np.ndarray)->dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred=model.predict(x_test)
        y_pred_prob=model.predict_proba(x_test)[:,1]

        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred_prob)

        metrics_dict={
            'accuracy':accuracy,
            'precsion':precision,
            'recall':recall,
            'auc':auc
        }
        logger.debug('model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('error during model evaluation: %s',e)
        raise

def save_metric(metrics:dict,file_path:str)->None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path ,'w') as f:
            json.dump(metrics,f,indent=4)
        logger.error('Metrics saved to %s',file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        model=load_model('./models/stacking_classifier.pkl')
        test_data=load_data('./data/processed/test_tfidf.csv')

        x_test=test_data.iloc[:,:-1].values
        y_test=test_data.iloc[:,-1].values

        metric=evaluate_model(model=model,x_test=x_test,y_test=y_test)

        # Experiment tracking using dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            live.log_params(params)

        save_metric(metric,'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")


if __name__=='__main__':
    main()


    