import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor()
            }

            param_grid = {
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5]
                },

                "Decision Tree": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10]
                },

                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5]
                },

                "XGBRegressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5, 7]
                },

                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 7, 9]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0]
                },

                "CatBoosting Regressor": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [100, 200]
                }
            }
            

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            # Best model find
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]
            logging.info(f"Best model: {best_model_name}")
            # Tuning Part 

            # 👉 Get param for best model
            params = param_grid.get(best_model_name, {})

            logging.info(f"\n🔧 Hyperparameter tuning started...")

            gs = GridSearchCV(
                        best_model,
                        params,
                        cv=3,
                        n_jobs=-1,
                        verbose=2   #  live progress
                    )

            gs.fit(X_train, y_train)

            logging.info(f"🔥 Best Params: {gs.best_params_}")
            logging.info(f"📈 CV Best Score: {gs.best_score_:.4f}")
            tuned_model = gs.best_estimator_
                    
            
            # 👉 Final evaluation with test data.
            y_pred = tuned_model.predict(X_test)
            final_score = r2_score(y_test, y_pred)

            logging.info(f"\n🎯 Final Score (after tuning): {final_score:.4f}")
    

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=tuned_model
                )
            

        
            return final_score

        except Exception as e:
            raise CustomException(e,sys)        