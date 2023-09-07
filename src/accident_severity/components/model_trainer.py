
import pandas as pd
import os, sys 
from pathlib import Path
from accident_severity.logging import logger
from xgboost import XGBClassifier
import joblib
from accident_severity.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config 

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop([self.config.target_column], axis=1)
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        y_test = test_data[[self.config.target_column]]

        xgb = XGBClassifier(n_estimators=self.config.n_estimators, max_depth=self.config.max_depth,
                                learning_rate=self.config.learning_rate)

        xgb.fit(X_train, y_train)

        joblib.dump(xgb, os.path.join(self.config.root_dir, self.config.model_name))
