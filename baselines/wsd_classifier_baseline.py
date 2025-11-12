import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.column].astype(str)


class FullContextCombiner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return (X['context_sentences'].astype(str) + ' ' + 
                X['ambiguous_sentence'].astype(str) + ' ' + 
                X['example_sentence'].astype(str))


# WSD-specific feature extractors
class WSDFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_columns = None
        self.vocab_before = set()
        self.vocab_after = set()
    
    def fit(self, X, y=None):
        # Learn vocabulary from training data
        for idx, row in X.iterrows():
            ambig_sent = str(row['ambiguous_sentence']).lower()
            homonym = str(row['homonym_word']).lower()
            
            if homonym in ambig_sent:
                parts = ambig_sent.split(homonym)
                if len(parts) >= 2:
                    before_words = parts[0].strip().split()[-3:]
                    after_words = parts[1].strip().split()[:3]
                    
                    for i, word in enumerate(before_words):
                        self.vocab_before.add(f'before_{i}_{word}')
                    for i, word in enumerate(after_words):
                        self.vocab_after.add(f'after_{i}_{word}')
        
        # Define fixed feature columns
        base_features = ['semantic_overlap', 'has_the', 'has_a', 'has_to', 
                        'has_of', 'ambig_length', 'context_length', 'example_overlap']
        self.feature_columns = base_features + sorted(list(self.vocab_before)) + sorted(list(self.vocab_after))
        
        return self
    
    def transform(self, X):
        features = []
        
        for idx, row in X.iterrows():
            feature_dict = {col: 0 for col in self.feature_columns}
            
            # 1. Context word features
            context = str(row['context_sentences']) + ' ' + str(row['ambiguous_sentence'])
            context_words = set(context.lower().split())
            
            # 2. Positional features - words before and after homonym
            ambig_sent = str(row['ambiguous_sentence']).lower()
            homonym = str(row['homonym_word']).lower()
            
            if homonym in ambig_sent:
                parts = ambig_sent.split(homonym)
                if len(parts) >= 2:
                    before_words = parts[0].strip().split()[-3:]
                    after_words = parts[1].strip().split()[:3]
                    
                    for i, word in enumerate(before_words):
                        feat_name = f'before_{i}_{word}'
                        if feat_name in feature_dict:
                            feature_dict[feat_name] = 1
                    for i, word in enumerate(after_words):
                        feat_name = f'after_{i}_{word}'
                        if feat_name in feature_dict:
                            feature_dict[feat_name] = 1
            
            # 3. Semantic similarity with judged meaning
            meaning_words = set(str(row['judged_meaning']).lower().split())
            overlap_with_context = len(meaning_words & context_words)
            feature_dict['semantic_overlap'] = overlap_with_context
            
            # 4. Syntactic features - POS-like patterns
            feature_dict['has_the'] = 1 if ' the ' in ambig_sent else 0
            feature_dict['has_a'] = 1 if ' a ' in ambig_sent else 0
            feature_dict['has_to'] = 1 if ' to ' in ambig_sent else 0
            feature_dict['has_of'] = 1 if ' of ' in ambig_sent else 0
            
            # 5. Sentence length features
            feature_dict['ambig_length'] = len(ambig_sent.split())
            feature_dict['context_length'] = len(context.split())
            
            # 6. Example sentence similarity
            example_words = set(str(row['example_sentence']).lower().split())
            example_overlap = len(example_words & context_words)
            feature_dict['example_overlap'] = example_overlap
            
            features.append(feature_dict)
        
        # Convert to DataFrame with fixed columns
        df_features = pd.DataFrame(features, columns=self.feature_columns)
        return df_features.values  # Return numpy array for sklearn compatibility


class ContextWindowExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.vectorizer = None
    
    def fit(self, X, y=None):
        contexts = self._extract_contexts(X)
        self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        self.vectorizer.fit(contexts)
        return self
    
    def transform(self, X):
        contexts = self._extract_contexts(X)
        return self.vectorizer.transform(contexts).toarray()  # Return dense array
    
    def _extract_contexts(self, X):
        contexts = []
        for idx, row in X.iterrows():
            sent = str(row['ambiguous_sentence']).lower()
            homonym = str(row['homonym_word']).lower()
            
            # Extract window around homonym
            words = sent.split()
            try:
                # Find first occurrence of homonym
                homonym_idx = next(i for i, w in enumerate(words) if homonym in w.lower())
                start = max(0, homonym_idx - self.window_size)
                end = min(len(words), homonym_idx + self.window_size + 1)
                context_window = ' '.join(words[start:end])
            except (ValueError, StopIteration):
                context_window = sent
            
            contexts.append(context_window)
        
        return contexts


class SemanticFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.meaning_vectorizer = None
    
    def fit(self, X, y=None):
        meanings = X['judged_meaning'].astype(str)
        self.meaning_vectorizer = TfidfVectorizer(max_features=50)
        self.meaning_vectorizer.fit(meanings)
        return self
    
    def transform(self, X):
        meanings = X['judged_meaning'].astype(str)
        return self.meaning_vectorizer.transform(meanings).toarray()  # Return dense array


class WSDClassifier:
    def __init__(self, model_type='svm', window_size=5):
        self.model_type = model_type
        self.window_size = window_size
        self.model = None
        
    def _create_feature_pipeline(self):
        
        feature_union = FeatureUnion([
            # WSD-specific features (collocations, overlaps, etc.)
            ('wsd_features', WSDFeatureExtractor()),
            
            # Context window around homonym
            ('context_window', ContextWindowExtractor(window_size=self.window_size)),
            
            # Semantic features from judged meaning
            ('semantic_features', SemanticFeatureExtractor()),
            
            # Full context TF-IDF (broader context)
            ('full_context', Pipeline([
                ('combiner', FullContextCombiner()),
                ('tfidf', TfidfVectorizer(max_features=150, ngram_range=(1, 3)))
            ])),
            
            # Ending sentence features (continuation context)
            ('ending_context', Pipeline([
                ('extractor', ColumnExtractor('ending_sentence')),
                ('tfidf', TfidfVectorizer(max_features=50))
            ]))
        ], transformer_weights={
            'wsd_features': 1.0,
            'context_window': 1.0,
            'semantic_features': 1.0,
            'full_context': 0.8,
            'ending_context': 0.5
        })
        
        return feature_union
    
    def _create_model(self):
        if self.model_type == 'svm':
            # SVM is traditionally strong for WSD
            return SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def fit(self, X, y):
        feature_pipeline = self._create_feature_pipeline()
        base_model = self._create_model()
        
        self.model = Pipeline([
            ('features', feature_pipeline),
            ('classifier', base_model)
        ])
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.model.named_steps['classifier'], 'predict_proba'):
            feature_pipeline = self.model.named_steps['features']
            classifier = self.model.named_steps['classifier']
            return classifier.predict_proba(feature_pipeline.transform(X))
        else:
            raise AttributeError("Model does not support probability prediction")
    
    def evaluate(self, X_test, y_test, verbose=True):
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate off-by-one accuracy (important for ordinal classes)
        off_by_one = np.sum(np.abs(y_test - y_pred) <= 1) / len(y_test)
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(y_test - y_pred))
        
        results = {
            'accuracy': accuracy,
            'off_by_one_accuracy': off_by_one,
            'mean_absolute_error': mae,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': conf_matrix
        }
        
        return results
    
    def analyze_feature_importance(self, X, top_n=20):
        if self.model_type in ['random_forest', 'gradient_boosting']:
            classifier = self.model.named_steps['classifier']
            feature_importance = classifier.feature_importances_
            
            # Get feature names from pipeline
            feature_names = self._get_feature_names()
            
            # Sort by importance
            indices = np.argsort(feature_importance)[::-1][:top_n]
            
            print(f"\nTop {top_n} Most Important Features:")
            for i, idx in enumerate(indices):
                if idx < len(feature_names):
                    print(f"{i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
        else:
            print("Feature importance only available for tree-based models.")
    
    def _get_feature_names(self):
        feature_names = []
        # This is a simplified version - full implementation would be more complex
        return [f"feature_{i}" for i in range(100)]

def run(train_file, test_file, scorer):
  
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
        'homonym_word': homonym_words * 3,
        'context_sentences': context_sentences * 3,
        'ambiguous_sentence': ambigous_sentences * 3,
        'judged_meaning': judged_meanings * 3,
        'ending_sentence': ending_sentences * 3,
        'example_sentence': example_sentences * 3,
        'score': scores * 3
    }
    
    df = pd.DataFrame(data)
    
    # Split data
    X_train = df.drop('score', axis=1)
    y_train = df['score']
    
    # Train WSD classifier with SVM (traditional WSD approach)
    wsd_clf = WSDClassifier(model_type='svm', window_size=5)
    wsd_clf.fit(X_train, y_train)
    
    # test set
    with open(test_file, 'r', encoding='utf8') as f:
        file_dict = json.load(f)

    ambigous_sentences, homonym_words, context_sentences, judged_meanings, ending_sentences, example_sentences, scores = [], [], [], [], [], [], []
    std_dev, avg = [], []
    for item in file_dict.values():
        homonym_words.append(item['homonym'])
        context_sentences.append(item['precontext'])
        judged_meanings.append(item['judged_meaning'])
        ending_sentences.append(item['ending'])
        example_sentences.append(item['example_sentence'])
        ambigous_sentences.append(item['sentence'])
        scores.append(item['choices'][scorer])
        std_dev.append(item['stdev'])
        avg.append(item['average'])
    
    data = {
        'homonym_word': homonym_words * 3,
        'context_sentences': context_sentences * 3,
        'ambiguous_sentence': ambigous_sentences * 3,
        'judged_meaning': judged_meanings * 3,
        'ending_sentence': ending_sentences * 3,
        'example_sentence': example_sentences * 3,
        'score': scores * 3,
        'avg': avg * 3,
        'stdev': std_dev * 3
    }
    
    df = pd.DataFrame(data)
    
    # Split data
    X_test = df.drop('score', axis=1)
    y_test = df['score']
    
    predictions = wsd_clf.predict(X_test)
    with open(f"predictions/wsd_classifier_{scorer}_predictions.JSONL", "a", encoding='utf8') as outfile:
        idx = 0
        for id in file_dict.keys():
            entry = {"id": id, "prediction": int(predictions[idx])}
            idx += 1
            outfile.write(json.dumps(entry) + "\n")


def main():
    train_file = 'data/train.json'
    dev_file = 'data/dev.json'
    
    # best p-value
    print("scorer 0")
    run(train_file, dev_file, 0)
    print()
    
    # decent accuracy and
    print("scorer 1")
    run(train_file, dev_file, 1)
    print()
    
    # print("scorer 2")
    # run(train_file, dev_file, 2)
    # print()
    
    # best accuracy
    print("scorer 3")
    run(train_file, dev_file, 3)
    print()
    
    # print("scorer 4")
    # run(train_file, dev_file, 4)
    # print()


if __name__ == "__main__":
    main()
