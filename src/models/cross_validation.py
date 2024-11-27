import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, SGDRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import os
from itertools import product
from sklearn.preprocessing import StandardScaler

class CrossValidation():
    def __init__(self,):
        
        ## training data
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        
        # features
        self.pca = [True, False]
        self.full_or_chunk = ['full'] #['full', 'chunked']
        self.embedding_type = ['class', 'final_output','mean']

        # comment out models as we go
        self.models = {
            # "RandomForest": RandomForestRegressor(),
            "SVC": SVR(),
            # "KNN": KNeighborsRegressor(),
            "SGD": SGDRegressor(),
            "XGBoost": XGBRegressor()
            }
        self.param_grids = {
            # "RandomForest": {'n_estimators': [25, 100, 250, 500], 'max_depth': [10, 20, 50], 'max_features': ['sqrt', 0.5]},
            "SVC": {'kernel': ['rbf'],'C': [0.55, 0.75, 1.0, 5.0,750], 'gamma': [1e-4, 1e-3, 5e-3, 0.01]},
            # "KNN": {'n_neighbors': [3, 5, 7, 11, 25]},
            
            # "SGD": {'alpha': [0.0001, 0.001, 0.01, 0.1], 'penalty': ['l2', 'l1'], 'loss': ['squared_error'], 'learning_rate': ['optimal'], 'max_iter': [1000]},
            "SGD": {
                    'alpha': [0.005, 0.01, 0.05, 0.1],
                    'penalty': ['l2', 'l1'],  # Keep both if interested in exploring sparsity
                    'loss': ['squared_error', 'huber', 'epsilon_insensitive'],  # Add alternative loss functions
                    'learning_rate': ['optimal', 'constant', 'invscaling'],  # Explore other learning rate strategies
                    'max_iter': [1000, 2000]  
                },
            "XGBoost": {'n_estimators': [250, 500, 750,1000], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 6, 7], 'gamma': [0, 0.1, 0.2]}
            }
       
        # saving
        self.output_path = "/home/ginger/code/gderiddershanghai/Group-201/experiments/BERT_RESULTS/"
        os.makedirs(self.output_path, exist_ok=True)

        
    def get_data(self, BERT_embedding_type, full_or_chunk, pca_status):
        
        def parse_embedding(row):
            # convert string back to floast
            return np.fromstring(row.strip("[]"), sep=" ")
        
        # reading in the data
        root = '/home/ginger/code/gderiddershanghai/Group-201/data/processed'
        train_fp = f'{root}/BERT/{BERT_embedding_type}/train/bert_{full_or_chunk}_train.csv'
        val_fp = f'{root}/BERT/{BERT_embedding_type}/val/bert_{full_or_chunk}_val.csv'        
        
        df_train = pd.read_csv(train_fp)
        df_val = pd.read_csv(val_fp)
        # print(df_val.head())
        self.y_train = np.array(df_train['rescaled_score'])
        self.y_val = np.array(df_val['rescaled_score'])
        
        # embeddigns were saved as strings
        if pca_status:
            self.X_train = np.array(df_train.iloc[:, -1].apply(parse_embedding).tolist())
            self.X_val = np.array(df_val.iloc[:, -1].apply(parse_embedding).tolist())
        else:
            self.X_train = np.array(df_train.iloc[:, -2].apply(parse_embedding).tolist())
            self.X_val = np.array(df_val.iloc[:, -2].apply(parse_embedding).tolist())

    def grid_search(self):
        i=0
        j=0
        # loading the correct data
        for BERT_embedding_type in self.embedding_type:
            for full_or_chunk in self.full_or_chunk:
                # if BERT_embedding_type=='mean': #and full_or_chunk=='chunked': 
                #     continue
                
                for pca_status in self.pca:
                    self.get_data(BERT_embedding_type, full_or_chunk, pca_status)
                    print(f'This will be the {i}th iteration of loading the data and the {j}th grid search')
                    i+=1
                    for model_name, model in self.models.items():
                        
                        if model_name in ['SVC', 'KNN', "SGD", "XGBoost", "LogisticRegression"]:
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(self.X_train)
                            X_val_scaled = scaler.transform(self.X_val)
                        else:
                            X_train_scaled = self.X_train
                            X_val_scaled = self.X_val
                        if model_name == "SVC" and BERT_embedding_type=='class' and pca_status: 
                            print('skipping - model_name == "SVC" and BERT_embedding_type==class and pca_status')
                            continue
                        
                        param_grid = self.param_grids[model_name]

                        # Generate all combinations of hyperparameters
                        param_combinations = list(product(*param_grid.values()))
                        param_names = list(param_grid.keys())

                        # set up saving
                        results_csv_path = os.path.join(self.output_path, f"grid_search_results_{model_name}.csv")

                        for params in param_combinations:
                            j+=1
                            param_dict = dict(zip(param_names, params))
                            
                            # remove gamma for linear kernel
                            # if model_name == "SVC" and param_dict.get('kernel') == 'linear':
                            #     param_dict = {k: v for k, v in param_dict.items() if k != 'gamma'}


                            try:
                                # SVM can throw errors
                                model.set_params(**param_dict)
                            except Exception as e:
                                print(f"Error setting parameters {param_dict} for model {model_name}: {e}")
                                continue

                            try:
                                # worried about xgboost
                                model.fit(X_train_scaled, self.y_train)
                            except Exception as e:
                                print(f"Error training model {model_name} with parameters {param_dict}: {e}")
                                continue
                            
                            
                            try:
                                # Predict and evaluate
                                y_pred = model.predict(X_val_scaled)
                                mse = mean_squared_error(self.y_val, y_pred)

                                # Save results

                                result_row = {
                                    'embedding_type': BERT_embedding_type,
                                    'full_or_chunk': full_or_chunk,
                                    'pca_status': pca_status,
                                    'model': model_name,
                                    'mse': mse,
                                    **param_dict
                                }
                                print(f"Model: {model_name}, Params: {param_dict}, MSE: {mse}")
                                # Check if the results CSV file already exists
                                if os.path.exists(results_csv_path):
                                    # Load the existing results CSV
                                    existing_df = pd.read_csv(results_csv_path)

                                    # Add any missing columns to the existing dataframe
                                    for key in result_row.keys():
                                        if key not in existing_df.columns:
                                            existing_df[key] = np.nan

                                    # Add any missing columns to the new result_row
                                    for column in existing_df.columns:
                                        if column not in result_row:
                                            result_row[column] = np.nan

                                    # Append the new row and save back
                                    updated_df = pd.concat([existing_df, pd.DataFrame([result_row])], ignore_index=True)
                                    updated_df.to_csv(results_csv_path, index=False)

                                else:
                                    # If the file does not exist, create it with the current row
                                    pd.DataFrame([result_row]).to_csv(results_csv_path, index=False)

                            except Exception as e:
                                print(f"Error predicting or saving results for model {model_name} with parameters {param_dict}: {e}")



                            #     result_row = {
                            #         'embedding_type': BERT_embedding_type,
                            #         'full_or_chunk': full_or_chunk,
                            #         'pca_status': pca_status,
                            #         'model': model_name,
                            #         'mse': mse,
                            #         **param_dict
                            #     }

                            #     if not os.path.exists(results_csv_path):
                            #         pd.DataFrame([result_row]).to_csv(results_csv_path, index=False)
                            #     else:
                            #         pd.DataFrame([result_row]).to_csv(results_csv_path, mode='a', header=False, index=False)

                            #     print(f"Model: {model_name}, Params: {param_dict}, MSE: {mse}")

                            # except Exception as e:
                            #     print(f"Error predicting or saving results for model {model_name} with parameters {param_dict}: {e}")
        
        
        
    # results_df = pd.DataFrame()