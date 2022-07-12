#Import needed packages
import sys
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This function reads in the messages and categories data then merges them into one dataframe
    
    Input:messages_filepath -Filepath to the disaster_messages.csv file for import
    categories_filepath - Filepath to the disaster_categories.csv file for import
    
    Output: df - Dataframe of the input data
    '''
    messages = pd.read_csv(messages_filepath, dtype = str)
    categories = pd.read_csv(categories_filepath, dtype = str)
    #Inner join dataframes on id
    df = messages.merge(categories, how = 'inner', on = ['id'])
    
    return df


def clean_data(df):
    '''
    This function cleans the catagory data for better processing
    
    Input: df - Dataframe with the messages and category data
    
    Output: df - Same dataframe as input but with cleaned category data    
    '''
    #Takes the categories column and splits on the semicolon to get individual categories
    categories = df['categories'].str.split(pat = ';', expand = True)
    #Only focus on the first row because we only need the category not the values
    row = categories.head(1)
    row_str = row.to_string(header = False, index = False)
    category_colnames = word_tokenize(row_str)
    #Drop the hyphen and the value at the end of each string
    category_colnames = [w[:-2] for w in category_colnames]
    #Replace column names with the category names
    categories.columns = category_colnames
    
    #Iterate through the categories
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = [int(val[-1]) for val in categories[column]]
    #Drop the old categories column
    df.drop(['categories'],axis =1, inplace = True)
    #Attach the cleaned category columns
    df = pd.concat([df, categories], axis = 1)
    #Remove any duplicate rows
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    This function saves the dataset to a sql database
    
    Input: df - Dataframe with all the clean data
    database_filename - Filename for the data in the sql database
    
    Output: data is saved in sql database
    '''
    update_filename_str = 'sqlite:///' + database_filename
    #Initialize the sql path
    engine = create_engine(update_filename_str)
    #Saves the file
    df.to_sql('CategorizedMessages', engine, index=False)


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
