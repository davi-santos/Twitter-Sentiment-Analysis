from my_funcions import *
from nltk.tokenize import SpaceTokenizer

start=time.time()

REMOVE_LINKS = 1
REMOVE_ARROBA = 2
REMOVE_PONCT  = 3

LIST_OF_TASKS = [REMOVE_LINKS, REMOVE_ARROBA, REMOVE_PONCT]

#textMSG='testo com vários @davi @vito @maria https://www.google.com/ e links'
textMSG = 'será que vai cortar @davi . ; esse texto ? https://www.google.com vamos ver né?'
print(remove_ponctuation(textMSG))


print("Texto=",textMSG)


tk = SpaceTokenizer()
#s1 = tk.tokenize(textMSG)
#print(s1)

for task in LIST_OF_TASKS:
    s1=tk.tokenize(textMSG)

    tam=len(s1)
    print("Num=",tam)

    #determine aqui o número de threads desejadas
    numthreads=10
    pedaco=int(tam/numthreads)
    threads=[]

    for i in range(0,numthreads):
        inicio=i*pedaco
        if i==numthreads-1:
            fim=tam
        else:
            fim=(i+1)*pedaco          
        
        textMSG=""
        
        for t in s1[inicio:fim]:
            textMSG=textMSG+t+" "

        thread = myThread(textMSG, task)
        thread.start()
        threads.append(thread)
        
    textMSG=""
    for t in threads:
        textMSG=textMSG+t.text
        t.join()
    print("Texto=",textMSG)

    