import numpy as np
import matplotlib.pyplot as plt

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
        calcoliamo l'energia cinetica
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

    #aggiornamento posizione e velocità con eulero
    def n_vel(self, fx, fy, dt):
        """
        date le componenti della forza e il passo temporale
        aggiorno le componenti della velocità
        """
        self.vx += fx*dt
        self.vy += fy*dt

    def n_pos(self, dt):
        """
        dato il passo temporale aggiorno le posizioni
        """
        self.x += self.vx*dt
        self.y += self.vy*dt



if __name__ == "__main__":

    g = 9.81 #acc di gravita' che vorrei tanto mettere uguale a 1
    p = Pallina(0, 10, 0, 0, 1)

    # simulazione moto
    N = 1500
    dt = 0.001

    #salvo le posizioni iniziali
    x = np.array([])
    y = np.array([])
    x = np.insert(x, len(x), p.x)
    y = np.insert(y, len(y), p.y)
    #array del tempo
    t = np.array([])
    t = np.insert(t, len(t), 0.0)

    #energia iniziale
    ene_cin = np.array([])
    ene_pot = np.array([])
    ene_cin = np.insert(ene_cin, len(ene_cin), p.energia_cinetica())
    ene_pot = np.insert(ene_pot, len(ene_pot), p.energia_potenziale_gravitazionale(g))


    for i in range(1, N):

        #se arrivi a terra fermati
        if y[-1]<0: break

        #moto di cadula libera
        fx = 0
        fy = -g

        #aggiorno le posizioni
        p.n_vel(fx, fy, dt)
        p.n_pos(dt)

        #salvo le posizioni
        x = np.insert(x, len(x), p.x)
        y = np.insert(y, len(y), p.y)

        #calcolo energia
        ene_cin = np.insert(ene_cin, len(ene_cin), p.energia_cinetica())
        ene_pot = np.insert(ene_pot, len(ene_pot), p.energia_potenziale_gravitazionale(g))

        t = np.insert(t, len(t), i*dt)


    #plot
    plt.figure(1)
    plt.title('moto parabolico')
    plt.xlabel('tempo')
    plt.ylabel('altezza')
    plt.plot(t, y)
    plt.grid()

    plt.figure(2)
    plt.title('energia moto parabolico')
    plt.xlabel('tempo')
    plt.ylabel('energia')
    plt.plot(t, ene_cin, label='energia cinetica')
    plt.plot(t, ene_pot, label='energia potenziale')
    plt.plot(t, ene_cin+ene_pot, label='energia totale')
    plt.legend(loc='best')
    plt.grid()

    plt.show()


