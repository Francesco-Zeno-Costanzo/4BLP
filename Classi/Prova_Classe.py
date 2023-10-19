import numpy as np

class Pallina:
    """
    Classe che rappresenta una pallina
    intesa come ogetto puntiforme
    """

    def __init__(self, x, y, vx, vy, m):
        """
        costruttore della classe, verra' chiamato
        quando creeremo l'istanza della classe
        in input prende la posizione, la velocita' e massa
        che sono le quantita' che identificano la pallina
        che saranno gli attributi della classe;
        il costruttore e' un particolare metodi della
        classe per questo si utilizzano gli underscore.
        il primo parametro che passiamo (self) rappresenta
        l'istanza della classe (self è un nome di default)
        questo perchè la classe è un modello generico che
        deve valere per ogni 'pallina'
        """
        #posizione
        self.x = x
        self.y = y
        #velocita'
        self.vx = vx
        self.vy = vy
        self.massa = m

    def energia_cinetica(self):
        """
        ad ogni metodo della classe viene passato
        come primo argomento self, quindi l'istanza
        calcoliamo l'evergia cinetica
        """
        #una volta creati gli attributi non e necessario passarli ai vari metodi
        ene_k = 0.5*self.massa*(self.vx **2 + self.vy**2)
        return ene_k

    def energia_potenziale_gravitazionale(self, g):
        """
        calcolimao l'energia potenziale; all'interno
        di un campo gravitazionale di intensita' g
        la nostra pallina ha energia per il semplice
        fatto di essere in un qualche punto del campo
        """

        #supponiamo g sia diretta verso il basso
        ene_u = self.massa*g*self.y
        return ene_u

g = 9.81 #acc di gravita' che vorrei tanto mettere uguale a 1

#creo l'istanza della classe
p1 = Pallina(1, 1, 2, 2, 1)
#chiamo i metodi sull'istanza
ene_tot_p1 = p1.energia_cinetica() + p1.energia_potenziale_gravitazionale(g)
print('energia pallina 1:', ene_tot_p1)

#creo l'istanza della classe
p2 = Pallina(2, 2, 2, 2, 1)
#chiamo i metodi sull'istanza
ene_tot_p2 = p2.energia_cinetica() + p2.energia_potenziale_gravitazionale(g)
print('energia pallina 2:',ene_tot_p2)