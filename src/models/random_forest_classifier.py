import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

class ChunkedCV:
    def __init__(self, df, chunk_col='chunk'):
        self.df = df
        self.chunk_col = chunk_col
        self.chunks = [df[df[self.chunk_col] == chunk].index.to_numpy() for chunk in df[self.chunk_col].unique()]

    def split(self):
        for i, chunk in enumerate(self.chunks):
            val_idx = self.chunks[i]
            train_idx = np.concatenate([self.chunks[j] for j in range(len(self.chunks)) if j != i])
            yield train_idx, val_idx

def grid_search_custom(df, model=RandomForestClassifier(), model_type='classifier', number_trees=[100, 250, 500], maximum_depths=[3, 5, 10]):
    df_results = pd.DataFrame()
    
    essays_to_sample = [1, 3, 4, 5, 6]
    df_subset = df[df['essay_set'].isin(essays_to_sample)]
    embedding_methods = ['mean', '[CLS]', 'pooler_output']
    scorer = make_scorer(cohen_kappa_score, weights="quadratic", greater_is_better=True)
    
    cv = ChunkedCV(df_subset, chunk_col='chunk')
    
    for embedding_method in embedding_methods:
        bert_extractor = BertFeatureExtractor()
        bert_embeddings = bert_extractor.transform(df_subset, embedding_type=embedding_method)
        
        for n_trees in number_trees:
            for max_depth in maximum_depths:
                model.set_params(n_estimators=n_trees, max_depth=max_depth)
                
                fold_results = []
                
                for train_idx, val_idx in cv.split():
                    X_train, X_val = bert_embeddings[train_idx], bert_embeddings[val_idx]
                    y_train, y_val = df_subset.loc[train_idx, 'rescaled_score'], df_subset.loc[val_idx, 'rescaled_score']
                    
                    model.fit(X_train, y_train)
                    score = scorer(model, X_val, y_val)
                    fold_results.append(score)
                
                avg_score = np.mean(fold_results)
                
                df_results = df_results.append({
                    'model_type': model_type,
                    'embedding_method': embedding_method,
                    'number_of_trees': n_trees,
                    'max_depth': max_depth,
                    'score': avg_score,
                    'classifier': str(model)
                }, ignore_index=True)
    
    df_results.to_csv('grid_search_results.csv', index=False)
    
    return df_results
