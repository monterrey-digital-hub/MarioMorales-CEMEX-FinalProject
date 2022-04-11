import pandas as pd
#import sqlite3 as db
#from sklearn.impute import SimpleImputer
import numpy as np
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, OrdinalEncoder, FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
# python
import pickle
#from functools import partial
# basics
#import scipy.stats as stats
# graphing
#import matplotlib.pyplot as plt
#import seaborn as sns
#import plotly.express as px
from sklearn.metrics import plot_confusion_matrix
# feature selection
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
# model selection
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve, make_scorer,
                             confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay)
# models
#from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
#from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
#from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier, XGBRegressor
#from imblearn.over_sampling import SMOTE
#from collections import Counter
from datetime import datetime

import unidecode
import string
import re
from names_dataset import NameDataset


def readdata():
    ISM_SourceData = pd.read_excel('Data/Tickets 2021-2022 (V02).xlsx', sheet_name="Squirrel SQL Export")
    return ISM_SourceData

def selectdata(ISM_SourceData, from_date, to_date):
    ISM_SourceData_test = ISM_SourceData[(ISM_SourceData['ACTUALSTART']>=from_date) & (ISM_SourceData['ACTUALSTART']<=to_date)]
    return ISM_SourceData_test

def cleandatastructure(ISM_SourceData_test, country_list):
    ## Put all text fields together
    ISM_SourceData_test['FULLDESC'] = ISM_SourceData_test['TITLE']+ ' ' +ISM_SourceData_test['LONGDESCRIPTION']
    ISM_SourceData_test['FULLDESC']
    ISM_test = ISM_SourceData_test[['TICKETID','FULLDESC','AffectedPersonCountry','OWNERGROUP']]

    ## Do some Cleaning
    ISM_test['FULLDESC'].fillna('NA',inplace = True)
    ISM_test['AffectedPersonCountry'].fillna('NA',inplace = True)
    ISM_test['FULLDESC'] = ISM_test['FULLDESC'].str.upper()

    #Delete Null values
    ISM_test = ISM_test[ISM_test['AffectedPersonCountry']!="NA"]
    ISM_test = ISM_test[ISM_test['FULLDESC']!="NA"]

    ## Consolidate Countriees
    ISM_test['AffectedPersonCountry'] = ISM_test['AffectedPersonCountry'].map({'DOM': 'Dominican Republic','Bahamas': 'Bahamas', 'Bangladesh': 'Banglasdesh',\
                                        'Belize': 'Belize','Central': 'Central', \
                                        'China': 'China','Clombia': 'Colombia','Colombia': 'Colombia','Costa Rica': 'Costa Rica','Croatia': 'Croatia','CZ': 'Czech Republic',\
                                        'Czech Republic': 'Czech Republic','DE': 'Germany','Dominican Republic': 'Dominican Republic', \
                                        'EG': 'Egypt','EGY': 'Egypt','Egypt': 'Egypt','El Salvador': 'El Salvador','ES': 'Spain','España': 'Spain',\
                                        'Filipinas': 'Philippines','FR': 'France','France': 'France', \
                                        'GB': 'United Kingdom','German': 'Germany','Germany': 'Germany','Great Britain': 'United Kingdom','Guatemala': 'Guatemala',\
                                        'Filipinas': 'Philippines',\
                                        'Haiti': 'Haiti','HU': 'Hungary','IL': 'Israel','Israel': 'Israel','Jamaica': 'Jamaica','JM': 'Jamaica','Malaysia': 'Malaysia',\
                                        'Latvia': 'Latvia','Mexic': 'Mexico',\
                                        'México': 'Mexico','Mexio': 'Mexico','Mexixo': 'Mexico','Mexico': 'Mexico','Filipinas': 'Philippines','MX': 'Mexico',\
                                        'Filipinas': 'Philippines','Netherlands': 'Netherlands',\
                                        'NIC': 'Nicaragua','Nicaragua': 'Nicaragua','Panam': 'Panama','Panama': 'Panama','Peru': 'Peru','Philippines': 'Philippines', \
                                        'PHL': 'Philippines','PH': 'Philippines','Philippines': 'Philippines','Poland': 'Poland','PL': 'Poland','Puerto Rico': 'Puerto Rico',\
                                        'Republica Dominicana': 'Dominican Republic','RO': 'Romania',\
                                        'Romania': 'Romania','Filipinas': 'Philippines','Republica Checca': 'Czech Republic','Spain': 'Spain','Switzerland': 'Switzerland',\
                                        'Trinidad y Tobago': 'Trinidad y Tobago', \
                                        'UAE': 'United Arab Emirates','UK': 'United Kingdom','United Kingdom': 'United Kingdom','United Arab Emirates': 'United Arab Emirates',\
                                        'Switzerland': 'Switzerland',\
                                        'United States of America': 'United States of America','United States of Americaśśś': 'United States of America',\
                                        'Unites States of America': 'United States of America',\
                                        'US': 'United States of America','USA': 'United States of America'})
    ISM_test['AffectedPersonCountry'].unique()

    ## Delete all not mapped values and not on country list
    ISM_test = ISM_test[ISM_test['AffectedPersonCountry'].isnull() == False]
    ISM_test = ISM_test[ISM_test['AffectedPersonCountry'].isin(country_list)]

    return ISM_test

def PrepareWordLists(datasets, df_words):
    # Recibe la lista de los datasets seccionados, para evitar tiempos muy largos de procesamiento 
    nd = NameDataset()
    no_name = nd.search('##CUCARAMACARATITIRIFUE##').get('first_name')
    for dataset in datasets:
        errcount = 0
        count = 0
        count_check = 1
        #print(dataset.info())

        for ind in dataset.index:
            #Seperar las palabras de cada regitro
            #Crear un registro para cada palabra
            count = count + 1
            if count == count_check:
                print(datetime.now())
                print(count)
                count_check = count_check + 500

            word_list = getlistofwords(dataset['FULLDESC'][ind], nd)
            word_list = list(dict.fromkeys(word_list))
            
            for word in word_list:  
                df_words = df_words.append({'COUNTRY':dataset['AffectedPersonCountry'][ind],'OWNERGROUP':dataset['OWNERGROUP'][ind],'WORDS':word.upper()}, ignore_index=True)
 
                            
    print(datetime.now())
    print('Words: ',len(df_words['WORDS'].unique())," Groups: ",len(df_words['OWNERGROUP'].unique()),' Countries:',len(df_words['COUNTRY'].unique()))
    return df_words


def buildwordsindex(ISM_wgwords_all, delete_limit, max_word_size):
    #Eliminar palabras que nod dan valor por pocas repeticiones.
    minimum_repeat = delete_limit
    CountedWords = pd.DataFrame(ISM_wgwords_all.groupby(['WORDS'])[['WORDS']].count())
    CountedWords = CountedWords.drop(CountedWords[CountedWords['WORDS'] < minimum_repeat].index)
    CountedWords['WORDS'] = CountedWords.index
    CountedWords.rename(columns ={'WORDS':'VALUES'}, inplace = True)

    ## Hacemos Inner Merge con ISM_wgwords para dejar solo las palabras útiles
    ISM_wgwords_all = pd.merge(ISM_wgwords_all, CountedWords, on="WORDS")
    ISM_wgwords_all = ISM_wgwords_all.drop(ISM_wgwords_all[ISM_wgwords_all['WORDS'].str.len()>25].index)
    
    ## Creamos columna Unique para identificar combinaciones únicas
    ISM_wgwords_all['UNIQUE'] = ISM_wgwords_all['COUNTRY'] + '<<|>>' + ISM_wgwords_all['OWNERGROUP']+ '<<|>>' + ISM_wgwords_all['WORDS']
    arr_wgwords_uniques = ISM_wgwords_all['UNIQUE'].unique()

    #Convertimos en dataframe y separamos las columnas
    df_uniquewords = pd.DataFrame(arr_wgwords_uniques, columns=['UNIQUE'])
    df_uniquewords[['COUNTRY','X1', 'OWNERGROUP', 'X2', 'WORDS']] = df_uniquewords['UNIQUE'].str.split('<<|>>', 5, expand=True)
    df_uniquewords = df_uniquewords[['COUNTRY', 'OWNERGROUP', 'WORDS']]

    #Guardamos los resultados en in dataframe = ISM_wgwords_uniques
    ISM_wgwords_uniques = df_uniquewords
    
    #Movemos los valores al índice para acelerar las búsquedas en la preparadación del trainin data sert
    ISM_wgwords_uniques['STR_INDEX']=ISM_wgwords_uniques['COUNTRY']+"-"+ISM_wgwords_uniques['OWNERGROUP']
    ISM_wgwords_uniques.index = ISM_wgwords_uniques['STR_INDEX']
    return ISM_wgwords_uniques


def trainingstructuredata(ISM_wgwords_uniques):
    GroupColumnsNames =ISM_wgwords_uniques['OWNERGROUP'].unique()
    TrainingData = pd.DataFrame(columns=GroupColumnsNames)
    TrainingData['COUNTRY'] = 'NA'
    TrainingData['OWNERGROUP'] = 'NA'
    TrainingData['TICKETID'] = 'NA'

    TrainingData.COUNTRY.datatype = object
    TrainingData.OWNERGROUP.datatype = object
    TrainingData.TICKETID.datatype = object
    return TrainingData



def training_dataset(ISM_test, TrainingData, ISM_wgwords_uniques, square_values):
## ISM_Test datos de ISM Country / Ownergroups / Full Text
## TrainingData Estructura de Salida
## ISM_wgwords_uniques: Calisficador de Palabras: Country / Ownergroup / Words
## square_values: Indica si se elevan al cuadrado los valores de matches de palabras (Bolean)
    nd = NameDataset()
    GroupColumnsNames =ISM_wgwords_uniques['OWNERGROUP'].unique()
    errcount = 0
    count = 0
    count2 = 0
    count_check = 1
    
    #ISM_test['FULLDESC_WORDS'] = getlistofwords_1(ISM_test['FULLDESC'], nd) -- Opcion pa optimizar
    
    for numindex, ind in enumerate(ISM_test.index):
        count = count + 1
        if count == count_check:
            print(datetime.now())
            print(count)

            if count == count_check:
                count_check = count_check + count_check

        word_list = getlistofwords(ISM_test['FULLDESC'][ind], nd)        
        #word_list = ISM_test['FULLDESC_WORDS'][ind] --> Opcion pa optimizar
        
        TrainingData = TrainingData.append({'TICKETID':ISM_test['TICKETID'][ind], 'COUNTRY':ISM_test['AffectedPersonCountry'][ind],'OWNERGROUP':ISM_test['OWNERGROUP'][ind]}, ignore_index=True) 
        #TrainingData = TrainingData.append([ISM_test['AffectedPersonCountry'][ind],ISM_test['OWNERGROUP'][ind]], ['COUNTRY','OWNERGROUP'],[numindex])
        
        ## Seteo valores de columnas de grupos del registros nuevo para que no sean null y puedan sumarse
        for column in TrainingData.columns:
            if column not in [0,'COUNTRY','OWNERGROUP','TICKETID']:
                TrainingData[column][numindex] = 0

        for columns in GroupColumnsNames:
            #grouplistwords = ISM_wgwords_uniques[(ISM_wgwords_uniques['COUNTRY']==ISM_test['AffectedPersonCountry'][ind]) &(ISM_wgwords_uniques['OWNERGROUP']==columns) ][['WORDS']]
            searchindex = ISM_test['AffectedPersonCountry'][ind]+"-"+columns
            
            try:
                grouplistwords = ISM_wgwords_uniques.loc[searchindex]
                grouplistwords = grouplistwords['WORDS'].to_list()

                list_matches = set(word_list) & set(grouplistwords)
                if len(list_matches) > 0:
                        #TrainingData[listwords['OWNERGROUP'][listidx]][numindex] = TrainingData[listwords['OWNERGROUP'][listidx]][numindex] + len(list_matches)
                    if square_values:
                        TrainingData[columns][numindex] = TrainingData[columns][numindex] + (len(list_matches))**2
                    else:
                        TrainingData[columns][numindex] = TrainingData[columns][numindex] + len(list_matches)
            except:
                errcount = errcount + 1
            else:
                errcount = errcount + 1
            
            
    print('Finished: ', datetime.now(), ' Errors:',errcount)
    return TrainingData


def exclusionlist():
        
    exclusionlist = ['OCTOBER','OCTUBRE','NOVEMBER','NOVIEMBRE','MAY','TUESDAY','MONDAY','WEDNESDAY','LUNES','VIERNES','FRIDAY','JUEVES','0','MARTES',\
        'THURSDAY','DOMINGO','MIERCOLES','JULIO','NOV','OCT','MON','SEPTIEMBRE','SABADO','SEPTEMBER','JAN','DOM','SATURDAY','AGO','MAR','SAB','ABRIL',\
        'FEBRERO','JUNE','SUNDAY','MAYO','AUGUST','AGOSTO','DICIEMBRE','JULY','DECEMBER','THU','APRIL','JUE','FRI','MARZO','SEP','AUG','LUN','JANUARY','JUNIO',\
        'ENERO','TUE','MIE','SAT','JUL','JUN','APR','FEBRUARY','DIC','WED','SUN','MARCH','VIE','FEB',\
        'SCREENSHOT','THE','AND','ATTACHMENT','PLEASE','HELLO','FOR','CEMEX','SUBJECT','FAVOR','RE','CAUTION',\
        'REGARDS','SALUDOS','listener','FROM','SUBJECT','CC','TO','FW','0','BUT','DIA','APOYO','SENT','WITH','THAT','COMO','CAN',\
        'HELP','BUT','LES','SUS', 'ADJUNT','SENDER','RECEIVED', \
        'SENT:','GSC', 'THIS', 'DEL','POR','PARA','QUE','CON','WAS','ARE','NOT','THANKS','THANK','LAS','ESTE','E-MAIL','SAVED','LOS','LAS','ESTE', \
        'BUEN','BUENOS','QUEDO','BEST','ASUNTO','DEAR','GRACIAS','GRACIAS','UNA','SALUDOS','RESOLVED','ATENTA','CORDIALES',\
        'ADJUNTO','ADJUNTA','ADJUNTOS','ADJUNTAMOS','ADJUNTANDO','*ADJUNTO','ATTACHED','ATTACHMENTS','ATTACH','ATTACHING','ATTACHED','ATTACHEE', \
        'ADDRESS','DIRECCION',\
        'A','ANTE','ANTES','BAJO','CON','CONTRA','DE','DESDE','PARA','POR','SEGUN','SIN','SOBRE','TRAS','PERO','SAN','NOS','NAME','PUEDE','PUEDA', \
        'UN','UNOS','UNA','UNAS','UNO','MAS','ETC','DIV', 'DUDA', 'SON' \
        'HOLA','BUEN','BUENAS','DIA','BUENOS','BUENA','DIAS','GRACIAS','MUCHAS','TARDES','SIGUIENTE','SIGUIENTES', 'PODRIAN',\
         'ESTA','ESTO','ESTAS','ESTOS','ESA','ESOS','ESAS','ESOS',\
         'AYUDA','TIENE','S/N','DEBE','RESPUESTA', 'TAMBIEN','SIGUE','FUE',\
         'HTTP','HTTPS','WWW','COM','MAILTO','COM/','JPG','CEL','NAME','TEL','TELEFONO',\
        'MAS', 'SOLICITUD','ATTENTION','REQUEST','TEAM','SERVICIO','TODOS', 'NOMBRE','N/A']

    return exclusionlist

def getlistofwords(str_desc, nd):
    no_name = nd.search('##CUCARAMACARATITIRIFUE##').get('first_name')
    errcount = 0
    shortest_word_list = []
    cleanstring = str_desc.replace("\\N"," ")
    cleanstring = cleanstring.replace(r"\N"," ")
    cleanstring = re.sub('<.*@.*>', ' ',cleanstring)
    cleanstring = re.sub('CR\d',' CRFOLIO ',cleanstring)
    cleanstring = re.sub('[-_"&.,\[\]?!¡#<>;:\'=$()*\\\]', ' ',cleanstring)
    
    #word_list = [re.sub('[?&,:;"><\'.()!¡\[\]-_]', ' ', unidecode.unidecode(wd.strip())) for wd in word_list]
    word_list = cleanstring.split(" ")
    word_list = [unidecode.unidecode(wd.strip()) for wd in word_list]
    
    short_word_list = set(word_list) - set(exclusionlist())
    shorter_word_list=[wd for wd in short_word_list if len(wd)>2]
    
    for word in shorter_word_list:
        #Evalua cada palabra y si no es de las exlcluidas por default la agrega a la lista de los palabars del grupo
        if word.find('CID') == -1 and word.find('@CEMEX.COM') == -1 and word.find('@EXT.CEMEX.COM') == -1 and word.find('IMAGE') == -1 and word.find('>>;') == -1 \
                    and word.find('GSC') == -1 and word.find('______') == -1 and word.find('---') == -1 and word.find('  ') == -1 and word.find('GIF') == -1 \
                    and word.find('SAFELINKS') == -1 and word.find('ATTACHE') == -1 and word.find('ADDRESS') == -1 \
                    and word.find('PNG') == -1 and word.find('E0') == -1 and word.find('E-') == -1 and word.find('****') == -1 and word.find('#') == -1 and word.find('LISTENER') \
                    and word.find('YOU') == -1 and word.find('@') == -1 and not word.startswith('ATT00') and not word.startswith('/')\
                    and not word.startswith('+') and not word[0].isdigit() and not word[1].isdigit() and not word.startswith('<') \
                    and not word.startswith('/>') and not word.startswith('@') and not word.startswith('?') and not word.startswith('"') and not word.startswith('|'):
                
            try:
                is_name = nd.search(word).get('first_name').get('country').get('Mexico')
                if is_name == no_name:
                    shortest_word_list.append(word)
            except:
                    errcount = errcount + 1
                    shortest_word_list.append(word)
    
    return shortest_word_list


def Create_training_data(ISM_Training_load, split_test_size):
    ##Convertimos los grupos en clasificaicones numéricas

    ownergroup_mapping = pd.DataFrame(ISM_Training_load['OWNERGROUP'].unique(), columns=['OWNERGROUP'])
    ownergroup_mapping['MAP_VALUE'] = ownergroup_mapping.index

    ## Crea el Valor numérico del Y (solver ownergroup)
    ISM_Training_load_full = pd.merge(ISM_Training_load, ownergroup_mapping, on='OWNERGROUP')
    ISM_Training_load_full.head(3)

    ## Construimos X y Y

    X_Train_Data = ISM_Training_load_full.drop(['OWNERGROUP', 'MAP_VALUE'], axis=1)
    Y_Train_Data = ISM_Training_load_full[['TICKETID','MAP_VALUE']]
    Y_Train_Data = Y_Train_Data.set_index('TICKETID') 

    ##Split Trainng Data 
    ## Tenemos que hacer el encoder antes, por el tema de los países, pueden salir columnas diferentes
    ## Al productizar, tenemos que creear una matriz de países y que sa la estructura base e ignorar el resto.

    datatrans_encoder = ColumnTransformer([
        ('one_hot', OneHotEncoder(handle_unknown='ignore'), ['COUNTRY'])
    ], remainder='passthrough', verbose_feature_names_out=False)

    df_temp = datatrans_encoder.fit_transform(X_Train_Data)
    df_temp = pd.DataFrame(df_temp)
    df_temp.columns = datatrans_encoder.get_feature_names_out()

    X_train_Data_fixed = df_temp
    X_train_Data_fixed = X_train_Data_fixed.set_index('TICKETID') 
    X_train_Data_fixed.head(5)

    ## Hacemos Split de Datos

    X_train, X_test, y_train, y_test = train_test_split(X_train_Data_fixed, Y_Train_Data, test_size=split_test_size)

    ## Cambio tipo de datos a númerico para procesos de training de modelos
    X_train_fixed = X_train.apply(pd.to_numeric)
    X_test_fixed = X_test.apply(pd.to_numeric)


    return X_train_fixed, X_test_fixed, y_train, y_test, ownergroup_mapping


def predictstructuredata(ISM_wgwords_uniques):
    GroupColumnsNames =ISM_wgwords_uniques['OWNERGROUP'].unique()
    TrainingData = pd.DataFrame(columns=GroupColumnsNames)
    TrainingData['COUNTRY'] = 'NA'
    TrainingData['TICKETID'] = 'NA'

    TrainingData.COUNTRY.datatype = object
    TrainingData.TICKETID.datatype = object
    return TrainingData


def predict_dataset(df_InputData, predictstructuredata, ISM_wgwords_uniques, square_values, NameDataset):
## ISM_Test datos de ISM Country / Ownergroups / Full Text
## TrainingData Estructura de Salida
## ISM_wgwords_uniques: Calisficador de Palabras: Country / Ownergroup / Words
## square_values: Indica si se elevan al cuadrado los valores de matches de palabras (Bolean)
    nd = NameDataset
    GroupColumnsNames =ISM_wgwords_uniques['OWNERGROUP'].unique()
    errcount = 0
    count = 0
    count2 = 0
    count_check = 1
    
    #ISM_test['FULLDESC_WORDS'] = getlistofwords_1(ISM_test['FULLDESC'], nd) -- Opcion pa optimizar
    
    for numindex, ind in enumerate(df_InputData.index):
        count = count + 1
        if count == count_check:

            if count == count_check:
                count_check = count_check + count_check

        word_list = getlistofwords(df_InputData['FULLDESC'][ind], nd)        
        #word_list = ISM_test['FULLDESC_WORDS'][ind] --> Opcion pa optimizar
        
        predictstructuredata = predictstructuredata.append({'TICKETID':df_InputData['TICKETID'][ind], 'COUNTRY':df_InputData['AffectedPersonCountry'][ind]}, ignore_index=True) 
        #TrainingData = TrainingData.append([ISM_test['AffectedPersonCountry'][ind],ISM_test['OWNERGROUP'][ind]], ['COUNTRY','OWNERGROUP'],[numindex])
        
        ## Seteo valores de columnas de grupos del registros nuevo para que no sean null y puedan sumarse
        for column in predictstructuredata.columns:
            if column not in [0,'COUNTRY','OWNERGROUP','TICKETID']:
                predictstructuredata[column][numindex] = 0

        for columns in GroupColumnsNames:
            #grouplistwords = ISM_wgwords_uniques[(ISM_wgwords_uniques['COUNTRY']==ISM_test['AffectedPersonCountry'][ind]) &(ISM_wgwords_uniques['OWNERGROUP']==columns) ][['WORDS']]
            searchindex = df_InputData['AffectedPersonCountry'][ind]+"-"+columns
            
            try:
                grouplistwords = ISM_wgwords_uniques.loc[searchindex]
                grouplistwords = grouplistwords['WORDS'].to_list()

                list_matches = set(word_list) & set(grouplistwords)
                if len(list_matches) > 0:
                        #TrainingData[listwords['OWNERGROUP'][listidx]][numindex] = TrainingData[listwords['OWNERGROUP'][listidx]][numindex] + len(list_matches)
                    if square_values:
                        predictstructuredata[columns][numindex] = predictstructuredata[columns][numindex] + (len(list_matches))**2
                    else:
                        predictstructuredata[columns][numindex] = predictstructuredata[columns][numindex] + len(list_matches)
            except:
                errcount = errcount + 1
            else:
                errcount = errcount + 1
            
    return predictstructuredata

def Create_Predict_data(Predict_InputData):

    datatrans_encoder = ColumnTransformer([
        ('one_hot', OneHotEncoder(handle_unknown='ignore'), ['COUNTRY'])
    ], remainder='passthrough', verbose_feature_names_out=False)

    df_temp = datatrans_encoder.fit_transform(Predict_InputData)
    df_temp = pd.DataFrame(df_temp)
    df_temp.columns = datatrans_encoder.get_feature_names_out()

    Predict_InputData = df_temp.set_index('TICKETID') 


    ## Cambio tipo de datos a númerico para procesos de training de modelos
    Predict_InputData = Predict_InputData.apply(pd.to_numeric)

    return Predict_InputData

class Predictive_Model:
    def __init__(self, *values):
        self.values = values

    def fit(self, wordlists, groupmapping, predictivemodel, namesdataset):
        with open('ISM_wgwords_uniques', 'rb') as f:
            self.wgwords_uniques = pickle.load(f)
        with open('group_mapping', 'rb') as f:
            self.group_mapping = pickle.load(f)
        with open('XGBClass_model', 'rb') as f:
            self.predict_model = pickle.load(f)
    
        self.nd = namesdataset
        #self.wgwords_uniques = wordlists
        #self.group_mapping = groupmapping
        #self.predict_model = predictivemodel
        self.Predict_data = predictstructuredata(self.wgwords_uniques)
        self.InputData = pd.DataFrame(columns=['TICKETID','FULLDESC','AffectedPersonCountry'])
        
    def predict(self, ticket_id, fulldesc, input_country):
        
        fulldesc = fulldesc.upper()
        
        self.InputData = self.InputData.append({'TICKETID': ticket_id, 'FULLDESC': fulldesc, 'AffectedPersonCountry': input_country}, ignore_index = True) 

        square_values = True
        df_InputData = predict_dataset(self.InputData, self.Predict_data, self.wgwords_uniques, square_values, self.nd)
        Predict_InputData = Create_Predict_data(df_InputData)

        y_predicted = self.predict_model.predict(Predict_InputData)
        predicted_group = self.group_mapping[self.group_mapping['MAP_VALUE']==y_predicted[0]][['OWNERGROUP']]
        
        return predicted_group.iloc[0]['OWNERGROUP']

    
class Build_PredictData:
    def __init__(self, *values):
        self.values = values

    def fit(self, wordlists, groupmapping, namesdataset):
        self.nd = namesdataset
        self.wgwords_uniques = wordlists
        self.group_mapping = groupmapping
        self.Predict_data = predictstructuredata(self.wgwords_uniques)
        self.InputData = pd.DataFrame(columns=['TICKETID','FULLDESC','AffectedPersonCountry'])
        
    def transform(self, ticket_id, fulldesc, input_country):
        self.ticket_id = ticket_id
        self.input_country = input_country
        fulldesc = fulldesc.upper()
        self.InputData = self.InputData.append({'TICKETID': ticket_id, 'FULLDESC': fulldesc, 'AffectedPersonCountry': input_country}, ignore_index = True) 

        square_values = True
        df_InputData = predict_dataset(self.InputData, self.Predict_data, self.wgwords_uniques, square_values, self.nd)
        Predict_InputData = Create_Predict_data(df_InputData)
        
        return Predict_InputData
    
    def real_value(self,y_predicted):
        predicted_group = self.group_mapping[self.group_mapping['MAP_VALUE']==y_predicted[0]][['OWNERGROUP']]
        
        return predicted_group.iloc[0]['OWNERGROUP'], self.ticket_id, self.input_country