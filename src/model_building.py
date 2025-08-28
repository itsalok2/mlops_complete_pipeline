import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df=pd.read_csv(file_path)
        logger.debug('data loaded from %s with shape %s',file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse the csv file: %s',e)
        raise
    except FileNotFoundError as e:
        logger.error('file nor found: %s',e)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loading the file %s',e)
        raise

def train_model(x_train:np.ndarray,y_train:np.ndarray)->StackingClassifier:
    """
    Train a StackingClassifier model with multiple base learners and a Logistic Regression
    as the meta-learner.

    :param x_train: Training features
    :param y_train: Training labels
    :return: Trained StackingClassifier
    """
    
    try:
        if x_train.shape[0]!=y_train.shape[0]:
            raise ValueError('number of sample in x_train and y_train must be same')
        
        logger.debug('Initializing StackingClassifier with base learners')

        svc = SVC(kernel= "sigmoid", gamma  = 1.0)
        knc = KNeighborsClassifier()
        mnb = MultinomialNB()
        dtc = DecisionTreeClassifier(max_depth = 5)
        lrc = LogisticRegression(solver = 'liblinear', penalty = 'l1')
        rfc = RandomForestClassifier(n_estimators = 50, random_state = 2 )
        abc = AdaBoostClassifier(n_estimators = 50, random_state = 2)
        bc = BaggingClassifier(n_estimators = 50, random_state = 2)
        etc = ExtraTreesClassifier(n_estimators = 50, random_state = 2)
        gbdt = GradientBoostingClassifier(n_estimators = 50, random_state = 2)    
        xgb  = XGBClassifier(n_estimators = 50, random_state = 2)
        
        logger.debug('Model training started with %d samples', x_train.shape[0])
        stacking_clf = StackingClassifier(
                estimators=[
                ('svc', svc),
                ('knc', knc),
                ('mnb', mnb),
                ('dtc', dtc),
                ('lrc', lrc),
                ('rfc', rfc),
                ('abc', abc),
                ('bc', bc),
                ('etc', etc),
                ('gbdt', gbdt),
                ('xgb', xgb)
            ],
            final_estimator=LogisticRegression(),
            passthrough=False
        )
        logger.debug('Model training started with %d samples', x_train.shape[0])
        stacking_clf.fit(x_train, y_train) 

        logger.debug('model training completed')
        return stacking_clf
    
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as f:
            pickle.dump(model,f)
        logger.debug('model saved to %s',file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    """
    Main execution function for the training pipeline.
    
    Steps:
    1. Load the preprocessed training data from CSV.
    2. Prepare features (X) and labels (y).
    3. Train a StackingClassifier model with multiple base learners and a Logistic Regression meta-learner.
    4. Save the trained model to disk for later use.
    """
    try:
        train_data = load_data('./data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        meta_learner = train_model(x_train, y_train)
        model_save_path = 'models/stacking_classifier.pkl'
        save_model(meta_learner, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the stacking model training pipeline: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()