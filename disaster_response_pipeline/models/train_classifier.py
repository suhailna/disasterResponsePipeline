"""
Classifier Trainer
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)

Sample Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>

Sample Script Execution:
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl

Arguments:
    1) Path to SQLite destination database (e.g. disaster_response_db.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""


import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV

def load_data(database_filepath):
   '''
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database (e.g. disaster_response_db.db)
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    '''
    
    # load data from the database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', con=engine)
    X = df.message.values 
    Y = df.iloc[:,4:]
    
    return X, Y, Y.columns


def tokenize(text):
    '''
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        # Remove the stop words
        if tok in stopwords.words("english"):
            continue
            
        # Reduce words to their stems
        tok = PorterStemmer().stem(tok)
        
        # Reduce words to their root
        tok = lemmatizer.lemmatize(tok).lower().strip()

        clean_tokens.append(tok)
    
    # Remove non alphabet characters
    clean_tokens = [tok for tok in clean_tokens if tok.isalpha()]
    return clean_tokens


def build_model():
    '''
    Build Pipeline function
    
    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.
    
    '''
    # Build the pipeline.
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Defining Parameters
    parameters = {
        'vect__max_df':[0.75,1.0],
        'clf__estimator__n_estimators': [20, 50]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)
    
    Arguments:
        
        model-> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names.
    
    '''
    Y_pred = model.predict(X_test)
    
    for ix, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:,ix]))

    
    acc = (Y_pred == Y_test).mean().mean()
    print("Accuracy Overall:")
    print(acc)


def save_model(model, model_filepath):
    '''
    
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        pipeline -> GridSearchCV or Scikit Pipelin object
        pickle_filepath -> destination path to save .pkl file
    
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()