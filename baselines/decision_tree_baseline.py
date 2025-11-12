import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor, export_text  # Using Regressor for scores
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.column]

class TextCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

class HomonymModel: 
    def __init__(self):
        self.model = None
        self.feature_pipeline = None
        
    def _create_feature_pipeline(self):
        feature_union = FeatureUnion([
            ('homonym', Pipeline([
                ('extract', ColumnExtractor('homonym_word')),
                ('tfidf', TfidfVectorizer(max_features=50, ngram_range=(1, 2)))
            ])),
            ('context', Pipeline([
                ('extract', ColumnExtractor('context_sentences')),
                ('tfidf', TfidfVectorizer(max_features=200, ngram_range=(1, 3)))
            ])),
            ('ambiguous', Pipeline([
                ('extract', ColumnExtractor('ambiguous_sentence')),
                ('tfidf', TfidfVectorizer(max_features=150, ngram_range=(1, 2)))
            ])),
            ('meaning', Pipeline([
                ('extract', ColumnExtractor('judged_meaning')),
                ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2)))
            ])),
            ('ending', Pipeline([
                ('extract', ColumnExtractor('ending_sentence')),
                ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2)))
            ])),
            ('example', Pipeline([
                ('extract', ColumnExtractor('example_sentence')),
                ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2)))
            ]))
        ])
        
        return feature_union
    
    def _create_model(self):
        return DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    
    def fit(self, X, y):
        self.feature_pipeline = self._create_feature_pipeline()
        
        base_model = self._create_model()
        
        self.model = Pipeline([
            ('features', self.feature_pipeline),
            ('regressor', base_model) 
        ])
        
        # Fit the model
        self.model.fit(X, y)
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
 

def run(train_file, test_file):
  
    with open(train_file, 'r', encoding='utf8') as f:
        file_dict = json.load(f)

    ambigous_sentences, homonym_words, context_sentences, judged_meanings, ending_sentences, example_sentences, scores = [], [], [], [], [], [], []
    
    for item in file_dict.values():
        homonym_words.append(item['homonym'])
        context_sentences.append(item['precontext'])
        judged_meanings.append(item['judged_meaning'])
        ending_sentences.append(item['ending'])
        example_sentences.append(item['example_sentence'])
        ambigous_sentences.append(item['sentence'])
        scores.append(np.mean(item['choices']))
    
    data = {
        'homonym_word': homonym_words,
        'context_sentences': context_sentences,
        'ambiguous_sentence': ambigous_sentences,
        'judged_meaning': judged_meanings,
        'ending_sentence': ending_sentences,
        'example_sentence': example_sentences,
        'score': scores
    }
    
    df = pd.DataFrame(data)
    
    X_train = df.drop('score', axis=1)
    Y_train = df['score']
      
    model = HomonymModel() 
    model.fit(X_train, Y_train)

    
    # test set
    with open(test_file, 'r', encoding='utf8') as f:
        file_dict = json.load(f)

    ambigous_sentences, homonym_words, context_sentences, judged_meanings, ending_sentences, example_sentences, scores = [], [], [], [], [], [], []
    
    for item in file_dict.values():
        homonym_words.append(item['homonym'])
        context_sentences.append(item['precontext'])
        judged_meanings.append(item['judged_meaning'])
        ending_sentences.append(item['ending'])
        example_sentences.append(item['example_sentence'])
        ambigous_sentences.append(item['sentence'])
        scores.append(np.mean(item['choices']))
    
    data = {
        'homonym_word': homonym_words,
        'context_sentences': context_sentences,
        'ambiguous_sentence': ambigous_sentences,
        'judged_meaning': judged_meanings,
        'ending_sentence': ending_sentences,
        'example_sentence': example_sentences,
        'score': scores
    }
    
    df = pd.DataFrame(data)
    
    # Split data
    X_test = df.drop('score', axis=1)
    y_test = df['score']
    
    predictions = model.predict(X_test)
    with open(f"predictions/decision_tree_predictions.JSONL", "a", encoding='utf8') as outfile:
        idx = 0
        for id in file_dict.keys():
            entry = {"id": id, "prediction": int(predictions[idx])}
            idx += 1
            outfile.write(json.dumps(entry) + "\n")

    
def main():
    train_file = './data/train.json'
    dev_file = './data/dev.json'
    
    print("Running Decision Tree Regressor")
    run(train_file, dev_file)
    print()
    
    

if __name__ == "__main__":
    main()