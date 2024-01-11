import time
import logging

# Creo il logger, __name__ è il nome del codice ed è __main__ se viene eseguito, __module__ se importato
logger = logging.getLogger("majg")#__name__)

# Settiamo il livello ad INFO, vogliamo controllare che vada tutto bene
logger.setLevel(logging.INFO)

# Settiamo il formato del messaggio di log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Creiamo il gestore del file
file_handler = logging.FileHandler('error.log', mode="w")
file_handler.setLevel(logging.ERROR) # Settiamo un livello diverso per il file 
file_handler.setFormatter(formatter) # sul file ci saranno solo i messaggi di errore

# Vogliamo vedere tutto anche su shell
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Aggiungiamo tutto al nostro logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

#================= main del codice =================

def divisione(x, y):
    """ Funzione particolarmente complicata
    """
    try :
        q = x / y
    
    except ZeroDivisionError:
        # logger.error restituisce solo il messaggio scritto da noi
        #logger.error('Qualcosa è andato storto si stava per verificare il second impact')
        
        # logger.exception ci resistuisce anche tutto il traceback
        logger.exception('Qualcosa è andato storto si stava per verificare il second impact')
    
    else :
        return q

# codice particolarmente complicato
for i, j in zip(range(15, 30), range(0, 15)):

    quoziente = divisione(i, j)

    logger.info(f"{i} / {j} = {quoziente}")
    time.sleep(1)
