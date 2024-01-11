import logging

# configuro il formato del logging; voglio sapere: il tempo, il nome del logger, il livello, e il messaggio
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def divisione(x, y):
    """ Funzione particolarmente complicata
    """
    return x / y

# codice particolarmente complicato
x_1 = 17
x_2 = 4

quoziente = divisione(x_1, x_2)

# controllo che tutto sia andato bene via print
print(f"{x_1} / {x_2} = {quoziente}")

# Controllo che tutto sia andato bene via logging, la stringa sarà il messaggio
logging.info(f"{x_1} / {x_2} = {quoziente}") 
