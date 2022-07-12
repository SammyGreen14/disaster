#Import needed packages
import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn import decomposition, datasets, tree
from sqlalchemy import create_engine
from sklearn.naive_bayes import MultinomialNB

def load_data(database_filepath):
    '''
    This function reads in the data from a sql database and outputs 2 lists for the messages and the categories as well as a dataframe of the       category values.
    
    Input: database_filepath - filepath of the sql database storing the cleaned data from process_data.py
    
    Output: X - List of the messages
    Y - Dataframe of the categories and values
    Labels - List of the the categories
    '''
    #Reads in the data from sql database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('CategorizedMessages', engine)
    #replace na with 0
    df = df.fillna(0)
    #Split the dataframe into response and explanatory variables
    X = df["message"].tolist()
    Y = df.drop(columns = ['message', 'original','id', 'genre']).values
    
    #Create the list of categories
    Labels = list(df.columns)
    Labels.remove('id')
    Labels.remove('message')
    Labels.remove('original')
    Labels.remove('genre')
    
    return X,Y,Labels


def tokenize(text):
    '''
    This function takes the messages as strings and tokenizes them. It also cleans the words for optimal processesing. 
    
    Input: text - string messages to be tokenized
    
    Output: clean_tokens - List of the clean tokens of the messages
    '''
    
    #Remove unwanted characters
    url_regex = 'http[s]://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(>:%[0-9a-fA-F][0-9a-fA-F]))+'
    #Break message into word tokens
    tokens = word_tokenize(text)
    #Initialize function and list used in for loop
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    #iterate through words
    for tok in tokens:
        #removes stopwords
        if tok not in stopwords.words("english"):
            #keep the root of the word
            stem = PorterStemmer().stem(tok)
            #keep the lowercase version with the base form of the word
            clean_tok = lemmatizer.lemmatize(stem).lower().strip()
            clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    '''
    This function uses a pipeline and gridsearch in order to build a model for the classification of emergency response messages.
    
    Input: None needed
    
    Output: tree_GS - optomized model
    '''
    
    dtreeCLF = tree.DecisionTreeClassifier()
    #Build pipeline to run TF-IDF as well as decision tree classifier with multiple outputs
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer = tokenize)),
        ('moc', MultiOutputClassifier(tree.DecisionTreeClassifier()))
        #('moc', MultiOutputClassifier(MultinomialNB()))

    ])
    
    #Assign values for parameters
    criterion = ["gini", "entropy"]
    max_depth = [2,6,10]
    
    #Set parameters for the grid search
    parameters = dict(moc__estimator__criterion=criterion,
                          moc__estimator__max_depth=max_depth)

    #run grid search to optomize the decision tree criterion and max depth at the same time
    tree_GS = GridSearchCV(pipeline, param_grid = parameters, verbose = 1)
    
    return tree_GS


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function evaluates the model created in the previous function to inform us if more tweaking is needed. 
    
    Input: model - Model built in previous function
    X_test - Dataframe of the test set for the messages
    Y_test - Dataframe of the test set for the category values
    category_names - List of the names of the 36 categories
    
    Output: Prints the scoring metrics for each category
    '''
    
  
    #print total model score
    print(model.score(X_test, Y_test))
    #Predict Y values for the X_test set
    y_pred = model.predict(X_test)
    #Ensure correct data type for classification report
    y_pred_df = pd.DataFrame(y_pred, columns = category_names)
    y_test_df = pd.DataFrame(Y_test, columns = category_names)
    #Iterate through the 36 categories
    for i in range(36):
        #for each category run the classification report to see how well the model is working
        print('Category: {}'.format(category_names[i]), "\n\n",
             classification_report(y_test_df.iloc[:,i], y_pred_df.iloc[:,i]))


def save_model(model, model_filepath):
    '''
    This function saves the model to a pickle file.
    
    Input: model - model created in prior function
    model_filepath - filepath that pickle file will be saved to 
    
    Output: File saved to path requested
    '''
    #Save the model to a pickle file
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
