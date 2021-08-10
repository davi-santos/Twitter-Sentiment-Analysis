from my_funcions import *
import datetime
#import numba

#VARIAVEIS
temp_petr_inicial = 0; temp_petr_final = 0
temp_mglu_inicial = 0; temp_mglu_final = 0
temp_irb_inicial = 0;  temp_irb_final = 0
temp_b3_inicial = 0;   temp_b3_final = 0

#LEITURA DOS DADOS

path_finances = './database/finance/' 
path_twitter =  './database/twitter/'

twitter_petr = pd.read_csv(path_twitter+'Petrobras.csv', index_col = 0)
twitter_mglu = pd.read_csv(path_twitter+'Magalu.csv', index_col = 0)
twitter_irb = pd.read_csv(path_twitter+'IRB_Brasil_Seguros.csv', index_col=0)
twitter_b3 = pd.read_csv(path_twitter+'B3.csv', index_col = 0)

#PRE PROCESSAMENTO
for i in range(1,11):
    tempo_preprocessamento_inicial = datetime.datetime.now()

    twitter_petr['text'] = twitter_petr['text'].apply(text_processing)
    twitter_mglu['text'] = twitter_mglu['text'].apply(text_processing)
    twitter_irb['text'] = twitter_irb['text'].apply(text_processing)
    twitter_b3['text'] = twitter_b3['text'].apply(text_processing)

    tempo_preprocessamento_final = datetime.datetime.now()
    time_diff = (tempo_preprocessamento_final-tempo_preprocessamento_inicial)
    execution_time = time_diff.total_seconds()
    print('Tempo de execucao preprocessamento em segundos de relogio %i: %.20lf' % (i, execution_time))