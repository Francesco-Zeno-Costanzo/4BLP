""" Codice per test id processi
"""
import os
import time
import multiprocessing as mp

def cube(x):
    ''' Funzione che calcola il cubo di x
    '''
    son_ID = os.getpid()
    dad_ID = os.getppid()
    print(f"son={son_ID}, dad={dad_ID}")
    time.sleep(1)
    return x**3


if __name__ == "__main__":

    start = time.time()
    # Uso Pool per creare i processi
    pool  = mp.Pool(processes=8)

    # Tramite map applico la funzione cube 
    # ad ogni elemento di range per ogni elemento di pool
    results = pool.map(cube, range(8))
    print(results)

    end = time.time() - start
    print(f"Elapsed time = {end}")
