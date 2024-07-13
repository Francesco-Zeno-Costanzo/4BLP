""" Hello world parallel
"""
import multiprocessing as mp

def f(x, name, output):
    '''
    funzione che "stampa" in parallelo hello world

    Parameters
    ----------
    x : int
        indice del processo
    name : string
        nome da stampare
    output : queue form multiprocessing
        memoria condivisa per conservare
        l'output di questa funzione 
    '''
    msg = f"Hello {name}"
    output.put((x, msg))

if __name__ == "__main__":
    # Coda degli output (Memoria condivisa)
    output = mp.Queue()

    # Creo una lista processi da eseguire, ne creiamo 4
    processes = [mp.Process(target=f, args=(x, "World", output)) for x in range(4)]

    # Eseguo i processi
    for p in processes:
        p.start()

    # Esco dai processi finiti
    for p in processes:
        p.join()

    # Estraggo i risultati dalla  coda degli output
    results = [output.get() for p in processes]

    print(results)