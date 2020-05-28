"""This module reads the dataset, cleans the data, and then stores it in a 
SQLite database
"""
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load messages and categories and merge them using the common id
    
    Args:
        messages_filepath (str): filepath of disaster_messages.csv
        categories_filepath (str): file path of disaster_categories.csv

    Returns:
        DataFrame: pandas dataframe contains both messages and categories 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """ The categories column contains combined 35 individual categories 
    1. split the the categories into individual categories
    2. set new column names
    3. encode the value of the individual categories
    4. drop duplicated rows
    
    Args:
        df (DataFrame): the dataframe that contains messages and categories
    
    Returns:
        DataFrame: cleaned dataframe with categories value being separated
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories to extract a list of new column names
    row = categories.iloc[0]
    category_colnames = row.str.slice(0,-2)
    categories.columns = category_colnames
    
    # Extract the last character of each string (the 1 or 0) as the new category value
    # For example, related-0 becomes 0, related-1 becomes 1
    # and convert the string to a numeric value.
    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = pd.to_numeric(categories[column])
    
    # Replace old categories column with 36 encoded individual categories
    df.drop(columns='categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """ Save the clean dataset into a sqlite database
    
    Args:
        df (DataFrame): the dataset in the form of dataframe
        database_filename (str): the database name to save dataset to
    Returns:
        None
    """
    table_name = 'cleaned_data'
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql(table_name, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()