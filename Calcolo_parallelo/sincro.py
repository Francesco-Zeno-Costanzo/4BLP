""" Codice per mostrare la sincornizzazione
"""
import multiprocessing as mp

def prelievo(bilancio, lock, N):
    ''' funzione che togle 1 ad una variabile nella shared memory N volte
    '''
    for i in range(N):
        lock.acquire()
        bilancio.value = bilancio.value - 1
        lock.release()

def deposito(bilancio, lock, N):
    ''' funzione che aggiunge 1 ad una variabile nella shared memory N volte
    '''
    for i in range(N):
        lock.acquire()
        bilancio.value = bilancio.value + 1
        lock.release()

def transizioni():
    # Variabile nella memoria condivisa, 'f' perche sia float
    # entrabi i processi possono accedervi e cambiarla
    bilancio = mp.Value('f', 100)

    # Se un processo vede che il lock è acquisito
    # non agisce finchè lo stesso non viene rilasciato
    lock = mp.Lock()

    # Cero i processi
    p1 = mp.Process(target=prelievo, args=(bilancio, lock, 10000))
    p2 = mp.Process(target=deposito, args=(bilancio, lock, 10000))

    # Faccio partire i processi
    p1.start()
    p2.start()

    # Aspetto la fine
    p1.join()
    p2.join()

    print(f"bilancio finale = {bilancio.value}")

if __name__ == "__main__":
    
    transizioni()