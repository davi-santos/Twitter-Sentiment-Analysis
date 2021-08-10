from joblib import parallel
import numpy as np
import pandas as pd
import nltk
nltk.download('rslp')
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
import re
import matplotlib.pyplot as plt


#MACHINE LEARNING FUNCTIONS FROM SKLEARN
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time, threading
from datetime import timedelta, datetime

RANDOM_STATE = 22

def remove_links(text):
    val = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return val

def remove_arroba(text):
    text = re.sub('@[^\s]+','',text)
    text = re.sub('#[^\s]+','',text)
    text = re.sub('$[^\s]+','',text)
    return text

def remove_ponctuation(text):
    val = re.sub(r'[^\w\s]',' ',text)
    return val

def remove_breaklines(text):
    text = text.split('\n')
    return ''.join(text)

def remove_numbers(text):
    text = re.sub(" \d+", ' ', text)
    return re.sub("\d+", ' ', text)

def extra_spaces(text):
    val = re.sub(' +',' ',text).lstrip().rstrip()
    return val

def stopwords_removal(text):
    words = nltk.corpus.stopwords.words('portuguese')
    text = [t for t in text.split() if t not in words]
    return ' '.join(text)

def stemming (text):
    stemmer = nltk.stem.RSLPStemmer()
    text = [stemmer.stem(t) for t in text.split()]
    return ' '.join(text)

def ruido_len1 (text):
    text = [t for t in text.split() if len(t) > 1]
    return ' '.join(text)

def text_processing(text):
    """
        Text processing steps:
        1. Lowering of text
        2. Remove Hyperlinks
        
        3. Remove metions such as @, # and $
        4. Remove breaklines \n
        5. Remove punctuation
        6. Remove numbers
    """
    #lowercase text
    text = text.lower()
    
    #remove hyperlinks
    text = remove_links(text)
    
    #remove @mentions, #hashtags and $tickers
    text = remove_arroba(text)
    
    #remove breaklines
    text = remove_breaklines(text)
    
    #remove ponctuation
    text = remove_ponctuation(text)
    
    #remove numbers
    text = remove_numbers(text)
    
    text = stopwords_removal(text)
    
    text = stemming(text)
    
    text = ruido_len1(text)
    
    #remove extra spaces
    text = extra_spaces(text)
    
    return text

class myThread (threading.Thread):
    text=""
    qual=0
    
    def __init__(self, t, q):
      threading.Thread.__init__(self)
      self.text=t
      self.qual=q
    
    #1
    def remove_links(self):
        val = re.sub(r'https?:\/\/.*[\r\n]*', '', self.text, flags=re.MULTILINE)
        self.text = val

    #2
    def remove_arroba(self):
        self.text = re.sub('@[^\s]+','',self.text)
        self.text = re.sub('#[^\s]+','',self.text)
        self.text = re.sub('$[^\s]+','',self.text)

    #3
    def remove_ponctuation(self):
        self.text=re.sub(r'[^\w\s]','',self.text)

    #4
    def remove_numbers(self):
        self.text=re.sub("\d+", '', self.text)
    
    #5
    def remove_breaklines(self):
        self.text = self.text.split('\n')
        self.text= ''.join(self.text)

    #6   
    def extra_spaces(self):
        val = re.sub(' +',' ',self.text).lstrip().rstrip()
        self.text=val

    #7
    def stopwords_removal(self):
        words = nltk.corpus.stopwords.words('portuguese')
        self.text = [t for t in self.text.split() if t not in words]
        self.text = ' '.join(self.text)

    #8
    def stemming (self):
        stemmer = nltk.stem.RSLPStemmer()
        self.text = [stemmer.stem(t) for t in self.text.split()]
        self.text= ' '.join(self.text)

    #9
    def ruido_len1 (self):
        self.text = [t for t in self.text.split() if len(t) > 1]
        self.text= ' '.join(self.text)   

    def run(self):
        if self.qual==1:
            self.remove_links()
        if self.qual==2:
            self.remove_arroba()
        if self.qual==3:
            self.remove_breaklines()
        if self.qual==4:
            self.remove_ponctuation()
        if self.qual==5:
            self.remove_numbers()
        if self.qual==6:
            self.stopwords_removal()
        if self.qual==7:
            self.stemming()

def sum_date(date_str, days=1):
    aux = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=days)
    return aux.strftime('%Y-%m-%d')

def create_dataframeTfidfTarget (df_twitter, df_finance, window_days=1, column=''):
    
    #Setting variables
    start_date = '2019-01-01'; end_date = '2019-05-31'
    flag = 0; mean_list = []; 

    final_dataframe = pd.DataFrame(columns='change'.split())

    #Rodar até data x ou flag==5
    while(flag < 5 or start_date!=end_date):
        
        #Dentro do loop se começa
        aux_day = start_date
        start_date = sum_date(start_date, 1)

        aux_vectors = pd.DataFrame(columns = [column])
        
        # Pegar vetores na janela de dias
        for j in range(window_days):
            aux_vectors = aux_vectors.append(df_twitter[df_twitter['data'] == aux_day])
            aux_day = sum_date(aux_day, 1)
        
        if aux_vectors.empty:
            continue
        
        mean_list.append(aux_vectors['text'].values.tolist())

        #Pegar data janela+1dia
        flag = 0
        
        while(df_finance[df_finance.index == aux_day].empty):
            flag +=1
            aux_day = sum_date(aux_day, 1)
            if flag == 10:
                mean_list.pop()
                break
        
        if flag != 10:
            final_dataframe = final_dataframe.append(df_finance[df_finance.index == aux_day])
    
    X = []; y = []
    aux = final_dataframe[column].values
    
    for i, idx in enumerate(mean_list):
        for j, jdx in enumerate(idx):
            X.append(jdx)
            y.append(aux[i])
            #aux = np.append(aux, aux[i])
            #y.append(aux)
    
    return X, y

def using_all_together(X_train, X_test, y_train, y_test, parallel=False):
    
    #Construindo Pipelines
                    
    pipe_lr = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                    ('clf', LogisticRegression(random_state=RANDOM_STATE, max_iter=100000))])

    pipe_rf = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                ('clf', RandomForestClassifier(random_state=RANDOM_STATE))])


    pipe_svm = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                ('clf', SVC(random_state=RANDOM_STATE))])

    # Parâmetros Grid Search
    param_range = [4, 8, 16]
    param_range_fl = [0.01, 1.0, 10, 100]

    grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
            'clf__C': param_range_fl,
            'clf__solver': ['liblinear']}] 


    grid_params_rf = [{'clf__criterion': ['gini'],
            'clf__max_depth': param_range,
            'clf__min_samples_split': param_range[1:]}]

    grid_params_svm = [{'clf__kernel': ['rbf'], 
            'clf__C': param_range_fl}]

    # Construindo Grid Search
    
    #TEMPO LR EM SEGUNDOS
    if parallel==True:
        LR = GridSearchCV(pipe_lr, param_grid=grid_params_lr, scoring='accuracy', cv=10, n_jobs=-1)
        RF = GridSearchCV(pipe_rf, param_grid=grid_params_rf, scoring='accuracy', cv=10, n_jobs=-1)
        SVM = GridSearchCV(pipe_svm, param_grid=grid_params_svm, scoring='accuracy', cv=10, n_jobs=-1)
    else:
        LR = GridSearchCV(pipe_lr, param_grid=grid_params_lr, scoring='accuracy', cv=10) 
        RF = GridSearchCV(pipe_rf, param_grid=grid_params_rf, scoring='accuracy', cv=10)
        SVM = GridSearchCV(pipe_svm, param_grid=grid_params_svm, scoring='accuracy', cv=10)

    # Lista de pipelines para iterar
    grids = [LR,RF,SVM]

    grid_dict = {0: 'Logistic Regression', 
            1: 'Random Forest',
            2: 'Support Vector Machine'}

    # Encontrar melhor estimador
    print('Executando pipelines com gridsearch...')
    best_acc = 0.0
    best_clf = 0
    best_gs = ''
    
    for idx, gs in enumerate(grids):
        
        #save initial time 0
        temp_init = datetime.now()
        #running grid search
        gs.fit(X_train, y_train)
        #save ending time 1
        temp_end = datetime.now()
        time_diff = (temp_end-temp_init)
        execution_time = time_diff.total_seconds()
        print('Tempo em segundos de relogio do %s: %.20lf' % (grid_dict[idx], execution_time))

        #print('\nClassificador: {}'.format(grid_dict[idx]))
        #print('Melhores parâmetros: {}'.format(gs.best_params_))
        #print('Acurácia treinamento {:.2}'.format(gs.best_score_))
        
        #Predicao do melhor gridsearch
        y_pred = gs.predict(X_test)
        
        # Test data accuracy of model with best params
        #print('Acurácia no teste: {:2.2}'.format(accuracy_score(y_test, y_pred)))
        #print('Confusion matrix: \n {}'.format(confusion_matrix(y_pred, y_test)))
        #print('Classification Report: \n {}'.format(classification_report(y_pred, y_test)))
        
        # Salvar modelo melhor acuracia
        if accuracy_score(y_test, y_pred) > best_acc:
            best_acc = accuracy_score(y_test, y_pred)
            best_gs = gs
            best_clf = idx
    #print('\nClassificador com melhor acurácia no teste: {:}'.format(grid_dict[best_clf]))