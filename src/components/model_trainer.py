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
from sklearn.model_selection import RandomizedSearchCV

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
                    "n_estimators": [50, 100, 200, 300, 500],
                    "max_depth": [None, 5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None],
                    "bootstrap": [True, False],
                    "criterion": ["squared_error", "absolute_error", "poisson"],
                },

                "Decision Tree": {
                    "max_depth": [3, 5, 10, 15, None],
                    "min_samples_split": [2, 5, 10, 20],
                    "min_samples_leaf": [1, 2, 4, 8],
                    "max_features": ["sqrt", "log2", None],
                    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    "splitter": ["best", "random"],
                    "ccp_alpha": [0.0, 0.01, 0.05, 0.1],
                },

                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "subsample": [0.6, 0.8, 1.0],
                    "max_features": ["sqrt", "log2", None],
                    "loss": ["squared_error", "absolute_error", "huber", "quantile"],
                },

                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False],
                },

                "XGBRegressor": {
                    "n_estimators": [50, 100, 200, 300, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "max_depth": [3, 5, 7, 9, 12],
                    "min_child_weight": [1, 3, 5, 7],
                    "gamma": [0, 0.1, 0.2, 0.5],
                    "subsample": [0.6, 0.7, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
                    "reg_alpha": [0, 0.01, 0.1, 1.0],
                    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
                    "scale_pos_weight": [1],
                },

                "CatBoosting Regressor": {
                    "iterations": [100, 200, 300, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "depth": [4, 6, 8, 10],
                    "l2_leaf_reg": [1, 3, 5, 7, 9],
                    "bagging_temperature": [0, 0.5, 1.0],
                    "border_count": [32, 64, 128],
                    "min_data_in_leaf": [1, 3, 5, 10],
                    "random_strength": [0.5, 1.0, 2.0],
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                    "loss": ["linear", "square", "exponential"],
                    "estimator": [
                        DecisionTreeRegressor(max_depth=1),
                        DecisionTreeRegressor(max_depth=2),
                        DecisionTreeRegressor(max_depth=3),
                    ],
                },

                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "leaf_size": [10, 20, 30, 40, 50],
                    "p": [1, 2],  # 1 = Manhattan, 2 = Euclidean
                    "metric": ["minkowski", "euclidean", "manhattan", "chebyshev"],
                },
            }
            
            # Step 1: Basic Evaluation
            model_report: dict =evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            # Step 2: Find the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            
            logging.info(f"Best model before tuning: {best_model_name} | Score: {best_model_score:.4f}")

            # ✅ Threshold check
            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable score.", sys)
        
            best_model = models[best_model_name]
            
            # # Step 3: Tuning Part 

            # 👉 Get param for best model
            params = param_grid.get(best_model_name, {})

            if params:
                logging.info(f"🔧 Tuning started for: {best_model_name}")

                gs = RandomizedSearchCV(
                            best_model,
                            params,
                            n_iter=50,          
                            cv=5,
                            scoring="r2",
                            n_jobs=-1,          
                            random_state=42,
                            verbose=2
                        )

                gs.fit(X_train, y_train)

                logging.info(f"🔥 Best Params: {gs.best_params_}")
                logging.info(f"📈 CV Best Score: {gs.best_score_:.4f}")
                tuned_model = gs.best_estimator_
            else:
                logging.info(f"⚡ No tuning needed for: {best_model_name}")
                tuned_model = best_model
                tuned_model.fit(X_train, y_train)                
                    
            
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