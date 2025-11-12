import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Custom transformer to extract specific columns
class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.column]

# Custom transformer to combine text columns
class TextCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

class HomonymClassifier:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_pipeline = None
        
    def _create_feature_pipeline(self):
        # Create separate TF-IDF vectorizers for different features
        feature_union = FeatureUnion([
            # Homonym word features
            ('homonym', Pipeline([
                ('extract', ColumnExtractor('homonym_word')),
                ('tfidf', TfidfVectorizer(max_features=50, ngram_range=(1, 2)))
            ])),
            
            # Context sentences features
            ('context', Pipeline([
                ('extract', ColumnExtractor('context_sentences')),
                ('tfidf', TfidfVectorizer(max_features=200, ngram_range=(1, 3)))
            ])),
            
            # Ambiguous sentence features
            ('ambiguous', Pipeline([
                ('extract', ColumnExtractor('ambiguous_sentence')),
                ('tfidf', TfidfVectorizer(max_features=150, ngram_range=(1, 2)))
            ])),
            
            # Judged meaning features
            ('meaning', Pipeline([
                ('extract', ColumnExtractor('judged_meaning')),
                ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2)))
            ])),
            
            # Ending sentence features
            ('ending', Pipeline([
                ('extract', ColumnExtractor('ending_sentence')),
                ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2)))
            ])),
            
            # Example sentence features
            ('example', Pipeline([
                ('extract', ColumnExtractor('example_sentence')),
                ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2)))
            ]))
        ])
        
        return feature_union
    
    def _create_model(self):
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def fit(self, X, y):
        # Create feature pipeline
        self.feature_pipeline = self._create_feature_pipeline()
        
        # Create model
        base_model = self._create_model()
        
        # Create full pipeline
        self.model = Pipeline([
            ('features', self.feature_pipeline),
            ('classifier', base_model)
        ])
        
        # Fit the model
        self.model.fit(X, y)
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.model.named_steps['classifier'], 'predict_proba'):
            return self.model.named_steps['classifier'].predict_proba(
                self.feature_pipeline.transform(X)
            )
        else:
            raise AttributeError("Model does not support probability prediction")
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return results

def run(train_file, test_file, model, scorer):
  
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
        scores.append(item['choices'][scorer])
    
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
    
    # Split data to features and target
    X_train = df.drop('score', axis=1)
    Y_train = df['score']
      
    # Train classifier
    clf = HomonymClassifier(model_type=model)
    clf.fit(X_train, Y_train)
    
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
        scores.append(item['choices'][scorer])
    
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
    
    predictions = clf.predict(X_test)
    with open(f"predictions/classifier_{model}_predictions.JSONL", "a", encoding='utf8') as outfile:
        idx = 0
        for id in file_dict.keys():
            entry = {"id": id, "prediction": int(predictions[idx])}
            idx += 1
            outfile.write(json.dumps(entry) + "\n")
    

def main():
    model_1 = 'random_forest'
    model_2 = 'gradient_boosting'
    model_3 = 'logistic_regression'
    
    train_file = 'data/train.json'
    dev_file = 'data/dev.json'
    
    # best performance on random forrest model - accuracy wise
    print(f"scorer number 1 on {model_1}")
    run(train_file, dev_file, model_1, scorer=1)
    print()
    
    # best performance on gradient boosting model - accuracy wise
    print(f"scorer number 3 on {model_2}")
    run(train_file, dev_file, model_2, scorer=3)
    print()
    
    # best performance on logistic regression model - accuracy wise
    print(f"scorer number 0 on {model_3}")
    run(train_file, dev_file, model_3, scorer=0)
    print()
    

if __name__ == "__main__":
    main()
