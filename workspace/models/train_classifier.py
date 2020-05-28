"""Build a classification model to predict 36 categories bases on the message input
"""

import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """Load data from database file path and extract features and labels

    Args:
        database_filepath (str): the path of database file

    Returns:
        X (2D array): message freatures to train model
        Y (2D array): output labels to learn from
        output_columns: 36 columns that contain individual categories 
    """
    # get the database name from database_filepath, the last string separated by "/"
#     dbname = database_filepath.split('/')[-1].strip()
    table_name = 'cleaned_data'
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(table_name, engine)
    X = df['message'].values
    output_columns = list(set(df.columns) - set({'id', 'message', 'original', 'genre'}))
    Y = df[output_columns].values
    return X, Y, output_columns


def tokenize(text):
    """Take the following steps to clean and tokenize text data:
        1. Replace any urls in text with "urlplaceholder" to reduce the data noise caused by url
        2. Split text into tokens.
        3. lemmatize, normalize case, and strip leading and trailing white space.
    Args:
        text (str): the input text to clean and tokenize
    Returns:
        list of string: cleaned and tokenized list of words
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()   
    return [lemmatizer.lemmatize(token).lower().strip() for token in tokens]


def build_model():
    """Build a model pipeline
    
    Returns:
        Pipeline: Model pipeline with the best parameters from cross validation grid search
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=50)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Report the f1 score, precision and recall for each output category of the dataset

    Args:
        model (obj): The trained model to evaluate
        X_test (2D array): the test features used for prediction
        Y_test (2D array): the test categories used to evaluate against test prediction
        category_names (list of string): the list of category names that model predicts

    Returns:
        None
    """
    Y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print(f"classification_report for category {category_names[i]}:")
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print(X.shape, Y.shape)
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