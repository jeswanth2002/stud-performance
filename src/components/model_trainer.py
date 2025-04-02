import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from src.custom_exception import CustomException
from src.custom_logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('resources','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array,test_array):
        try:
            logging.info("starting the model trainer")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "LinearRegression" : LinearRegression(),
                "DescisionTree" : DecisionTreeRegressor(),
                "RandomForest" : RandomForestRegressor(),
                "K-Neightbors" : KNeighborsRegressor(),
                "GradientBoosting" : GradientBoostingRegressor(),
                "AdaBoost" : AdaBoostRegressor()
            }
            model_report = evaluate_models(X_train, y_train, X_test, y_test,models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score < 0.5:
                raise CustomException("No best model found")

            logging.info(f"Best model found is {best_model_name} with score {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj=best_model
            )

            logging.info("Model training completed successfully")

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        except:
            pass