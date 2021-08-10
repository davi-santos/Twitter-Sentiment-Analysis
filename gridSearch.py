from my_funcions import *

#LEITURA DOS DADOS

path_finances = './database/finance/' 
path_twitter =  './database/twitter/'

#PETR4
petr4_financas = pd.read_csv(path_finances+'Petrobras.csv', index_col=0)
petr4_twitter = pd.read_csv(path_twitter+'/Petrobras.csv', index_col = 0)

#Magazine Luisa
mgu_financas = pd.read_csv(path_finances+'Magazine Luisa SA.csv', index_col=0)
mgu_twitter = pd.read_csv(path_twitter+'/Magalu.csv', index_col = 0)

#IRB Brasil Seguros
irb_financas = pd.read_csv(path_finances+'IRB Brasil Seguros.csv', index_col=0)
irb_twitter = pd.read_csv(path_twitter+'/IRB_Brasil_Seguros.csv', index_col=0)

#B3
b3_financas = pd.read_csv(path_finances+'B3.csv', index_col=0)
b3_twitter = pd.read_csv(path_twitter+'/B3.csv', index_col = 0)

#PRE-PROCESSAMENTO

petr4_twitter['text'] = petr4_twitter['text'].apply(text_processing)
petr4_twitter['data'] = petr4_twitter['date'].apply(lambda x: x.split()[0])
petr4_financas['change'] = petr4_financas['Close'].pct_change()
petr4_financas['Divide em 0.0'] = petr4_financas['change'].apply(lambda x: 0 if x < 0 else 1)

mgu_twitter['text'] = mgu_twitter['text'].apply(text_processing)
mgu_twitter['data'] = mgu_twitter['date'].apply(lambda x: x.split()[0])
mgu_financas['change'] = mgu_financas['Close'].pct_change()
mgu_financas['Divide em 0.0'] = mgu_financas['change'].apply(lambda x: 0 if x < 0 else 1)

irb_twitter['text'] = irb_twitter['text'].apply(text_processing)
irb_twitter['data'] = irb_twitter['date'].apply(lambda x: x.split()[0])
irb_financas['change'] = irb_financas['Close'].pct_change()
irb_financas['Divide em 0.0'] = irb_financas['change'].apply(lambda x: 0 if x < 0 else 1)

b3_twitter['text'] = b3_twitter['text'].apply(text_processing)
b3_twitter['data'] = b3_twitter['date'].apply(lambda x: x.split()[0])
b3_financas['change'] = b3_financas['Close'].pct_change()
b3_financas['Divide em 0.0'] = b3_financas['change'].apply(lambda x: 0 if x < 0 else 1)

#TREINAMENTO

#PETR4
X_petr4, y_petr4 = create_dataframeTfidfTarget(petr4_twitter, petr4_financas, window_days=1, column='Divide em 0.0')
X_train, X_test, y_train, y_test = train_test_split(X_petr4, y_petr4, stratify=y_petr4, test_size=0.3, random_state=RANDOM_STATE)
print('Teste 1: PETR4 SINGLE CORE')
using_all_together(X_train, X_test, y_train, y_test, parallel=False)
print('Teste 2: PETR4 PARALELO')
using_all_together(X_train, X_test, y_train, y_test, parallel=True)

#MGLU3
X_mglu3, y_mglu3 = create_dataframeTfidfTarget(mgu_twitter, mgu_financas, window_days=1, column='Divide em 0.0')
X_train, X_test, y_train, y_test = train_test_split(X_mglu3, y_mglu3, stratify=y_mglu3, test_size=0.3, random_state=RANDOM_STATE)
print('Teste 3: MGLU3 SINGLE CORE')
using_all_together(X_train, X_test, y_train, y_test, parallel=False)
print('Teste 4: MGLU3 PARALELO')
using_all_together(X_train, X_test, y_train, y_test, parallel=True)

'''
#B3
X_b3, y_b3 = create_dataframeTfidfTarget(b3_twitter, b3_financas, window_days=1, column='Divide em 0.0')
X_train, X_test, y_train, y_test = train_test_split(X_b3, y_b3, stratify=y_b3, test_size=0.3, random_state=RANDOM_STATE)
print('Teste 5: B3 SINGLE CORE')
using_all_together(X_train, X_test, y_train, y_test, parallel=False)
print('Teste 6: B3 PARALELO')
using_all_together(X_train, X_test, y_train, y_test, parallel=True)

#IRB
X_irb, y_irb = create_dataframeTfidfTarget(irb_twitter, irb_financas, window_days=1, column='Divide em 0.0')
X_train, X_test, y_train, y_test = train_test_split(X_irb, y_irb, stratify=y_irb, test_size=0.3, random_state=RANDOM_STATE)
print('Teste 5: IRB SINGLE CORE')
using_all_together(X_train, X_test, y_train, y_test, parallel=False)
print('Teste 6: IRB PARALELO')
using_all_together(X_train, X_test, y_train, y_test, parallel=True)
'''