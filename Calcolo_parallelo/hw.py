""" Hello world parallel
"""

import multiprocessing as mp

def f(name):
    print(f"Hello {name}")

if __name__ == "__main__":
    # Creo il processo passando la funzione da
    # parallelizzare e relativi argomenti
    p = mp.Process(target=f, args=("World",))
    # Parte l'esecuzione
    p.start()
    # Termina l'esecuzione
    p.join()
