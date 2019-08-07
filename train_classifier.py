import sys
import pandas as pd 
from sqlalchemy import create_engine
import re
import nltk

from os.path import basename
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    
    '''
    loads database from path and outputs X and y as numpy arrays + categories
    
    INPUT 
        database_filepath  - path to sqlite database 
    
    OUTPUT
        X - messages text array
        y - categories values ( 0 or 1) array
        category_names - name of each caregory contained in y 
    '''
    
    # database url and table name in the database 
    db_url = 'sqlite:///' + database_filepath
    db_table = basename(database_filepath).split(sep='.')[0]
    
    # load data from database
    engine = create_engine(db_url)
    sql="SELECT * FROM " + db_table
    df= pd.read_sql(sql, engine)
    
    X = df['message'].values 
    y=(df.drop(['id','message','original','genre'],axis=1)).values
    category_names = df.drop(['id','message','original','genre'],
                             axis=1).columns
    return X, y, category_names

def tokenize(text):
    
    '''
    takes in a string of text and tokenizes, cleans, and normalizes it 
    
    INPUT 
        text  - string to be tokenized 
    
    OUTPUT
        clean_tokens - a list of original text tokens 
    '''
    
    # regular expression to identify url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # replace any URL by"urlplaceholder"
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # create tokens from raw text, eliminate stop words, lemmatize each token, and clean the tokens
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stop_words]

    return clean_tokens


def build_model():
    
    '''
    builds an sklearn pipeline that transforms the data 
    and then applies a classifier  to it 
    tune the model using through cross-validation 
    '''
    # sklearn pipeline
    pipeline = Pipeline([
    ('tfidf_vect',TfidfVectorizer(tokenizer=tokenize)),
    ('clf',RandomForestClassifier())
    ])
    
    # list of parameters to tune using GridSearch  
    parameters = {
    'tfidf_vect__max_df':[0.5,1.0],
    'clf__n_estimators': [50, 200],
    'clf__min_samples_split': [2, 4]
    }

    # cross-validated classifier model 
    model = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1,cv=3)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    
    for i,label in enumerate(category_names):
    
        prediction = y_pred[:,i]
        print('results for:',label)
        print(classification_report(Y_test[:,i],prediction))
        accuracy = (prediction == Y_test[:,i]).mean()
        print("Accuracy:", accuracy)
        print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    # save the model to disk
    joblib.dump(model, model_filepath)
    
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        
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