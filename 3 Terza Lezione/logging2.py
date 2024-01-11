import time
import logging

# configuro il formato del logging; voglio sapere: il tempo, il nome del logger, il livello, e il messaggio
# Inoltre piuttosto che su shell deve essere stampato su un file, che deve essere sovrascritto ad ogni esecuzione del codice
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename="file.log", filemode="w") # filemode di default è a (append)

def divisione(x, y):
    """ Funzione particolarmente complicata
    """
    return x / y

# codice particolarmente complicato
for i, j in zip(range(15, 30), range(1, 16)):

    quoziente = divisione(i, j)

    # Controllo che tutto sia andato bene via logging, la stringa sarà il messaggio
    logging.info(f"{i} / {j} = {quoziente}")
    time.sleep(1)
