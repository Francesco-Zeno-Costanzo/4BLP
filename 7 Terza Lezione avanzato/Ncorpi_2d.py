import time
import numpy as np
import random as rn
import matplotlib.pyplot as plt
from matplotlib import animation

# For better plot
#import mplhep
#plt.style.use(mplhep.style.CMS) 

class Body:
    """
    Classe che rappresenta una pallina
    intesa come ogetto puntiforme
    """

    def __init__(self, x, y, vx, vy, m=1):
        """
        costruttore della classe, verra' chiamato
        quando creeremo l'istanza della classe, (i.e. b=Body()
        b e' chiamata istanza).
        In input prende la posizione, la velocita' e massa
        che sono le quantita' che identificano il corpo
        che saranno gli attributi della classe;
        il costruttore e' un particolare metodi della
        classe per questo si utilizzano gli underscore.
        il primo parametro che passiamo (self) rappresenta
        l'istanza della classe (self e' un nome di default)
        questo perche' la classe e' un modello generico che
        deve valere per ogni corpo.
        """
        #posizione
        self.x = x
        self.y = y
        #velocita'
        self.vx = vx
        self.vy = vy
        #massa
        self.m = m
    
    #aggiornamento posizione e velocita' con eulero
    def n_vel(self, fx, fy, dt):
        """
        ad ogni metodo della classe viene passato
        come primo argomento self , quindi l ' istanza
        date le componenti della forza e il passo temporale
        aggiorno le componenti della velocita'
        """
        self.vx += fx*dt
        self.vy += fy*dt

    def n_pos(self, dt):
        """
        dato il passo temporale aggiorno le posizioni
        """
        self.x += self.vx*dt
        self.y += self.vy*dt

class Sistema:
    '''
    Classe per evoluzione del sistema.
    Viene utilizzata la tecnica del softening  per impedire
    divergenze nella foza, sp e' il parametro di softening
    
    Parameters
    ----------
    corpi : list
        lista di ogetti della classe Body
    G : float
        Costante di gravitazione universale (=1)
    sp : float, optional, default 0
        parametro di softening
    '''

    def __init__(self, corpi, G, sp=0):
        self.corpi = corpi
        self.G = G
        self.sp = sp

    def evolvo(self, dt):
        '''
        chimata ad ogni passo temporale, fa evolvere il sistema
        solo di uno step dt, la forza e' calcolata secondo la
        legge di gravitazione universale;
        '''

        for corpo_1 in self.corpi:

            fx = 0.0
            fy = 0.0

            for corpo_2 in self.corpi:
                if corpo_1 != corpo_2:

                    dx = corpo_2.x - corpo_1.x
                    dy = corpo_2.y - corpo_1.y

                    d = np.sqrt(dx**2 + dy**2 + self.sp)

                    fx += self.G * corpo_2.m * dx / d**3
                    fy += self.G * corpo_2.m * dy / d**3

            corpo_1.n_vel(fx, fy, dt)

        for corpo in self.corpi:
            corpo.n_pos(dt)

class Measure:
    '''
    Parameters
    ----------
    bodies : list
        list of object from Body class
    G : float
        universal gravitational constant (=1)
    sp : float, optional, default 0
        softening parameter
    '''
    def __init__(self, bodies, G, sp=0):
        self.bodies = bodies # List of alla body
        self.G     = G       # 6.67x10^-11 = 1
        self.sp    = sp      # Softening parameter
    
    def energy(self):
        ''' Compute the total energy
        '''
        K = 0
        V = 0
        for body in self.bodies:
            K += 0.5*body.m*(body.vx**2 + body.vy**2)
        
        all_body = self.bodies.copy()
        for body_1 in all_body:
            for body_2 in all_body:
                if body_1 != body_2:
                    dx = body_2.x - body_1.x
                    dy = body_2.y - body_1.y
                    d = np.sqrt(dx**2 + dy**2 + self.sp)

                    V += -body_1.m*body_2.m*self.G/d
            all_body.remove(body_1)

        return K + V
    
    def angular(self):
        ''' Compute the total angular momentum
        '''
        L = 0
        for body in self.bodies:
            l_i = body.m*(body.x*body.vy - body.y*body.vx)
            L += l_i
        
        return L

#===========================================================================
# Creating bodies and the system and computational parameters
#===========================================================================

rn.seed(69420)
dt = 1/20000
T  = int(2/dt)
E  = np.zeros(T)
L  = np.zeros(T)
G  = 1

# Number of body, must be even
#N = 10
#C = []
#for n in range(N//2):
#    '''
#    two bodies are created at a time
#    with equal and opposite velocity
#    to keep the total momentum of the system zero
#    '''
#    v_x = rn.uniform(-0.5, 0.5)*0
#    v_y = rn.uniform(-0.5, 0.5)*0
#    C.append(Body(rn.uniform(-0.5, 0.5), rn.uniform(-0.5, 0.5), v_x, v_y))
#    C.append(Body(rn.uniform(-0.5, 0.5), rn.uniform(-0.5, 0.5), -v_x, -v_y))

# creation of body
C1 = Body(0.5,  0, 0, 20,  int(1e3))
C2 = Body(-0.5, 0, 0, -20, int(1e3))
C3 = Body(-1.5, 0, 0, 40,  int(1e1))
C  = [C1, C2, C3]
N  = len(C)
X  = np.zeros((2, T, N)) # 2 because the motion is on a plane

# Creation of the system
soft = 0.0
sist = Sistema(C, G, soft)
M    = Measure(C, G, soft)

#===========================================================================
# Evolution
#===========================================================================

start = time.time()

for t in range(T):

    L[t] = M.angular() # measure angular momentum
    E[t] = M.energy()  # measure energy

    sist.evolvo(dt)
    for n, body in enumerate(sist.corpi):
        X[:, t, n] = body.x, body.y

print("--- %s seconds ---" % (time.time() - start))

#===========================================================================
# Plot and animation
#===========================================================================
t = np.linspace(0, T*dt, T)
plt.figure(0)#, figsize=(10, 9))
plt.title('Energy of the system')#, fontsize=20)
plt.grid()
plt.plot(t, (E-E[0])/E)
plt.xlabel('t')#, fontsize=20)
plt.ylabel(r'$\frac{E(t)-E(t_0)}{E(t)}$')#, fontsize=20)
plt.ticklabel_format(axis='y', style='sci', scilimits = (0,0))
#plt.savefig("ene_euler_simpl.pdf")

plt.figure(1)#, figsize=(10, 9))
plt.title('Angular momentum')#, fontsize=20)
plt.grid()
plt.plot(t, (L -L[0])/L)
plt.xlabel('t')#, fontsize=20)
plt.ylabel(r'$\frac{L(t)-L(t_0)}{L(t)}$')#, fontsize=20)
#plt.savefig("ang_euler_simpl.pdf")

fig = plt.figure(2)
plt.grid()
plt.xlim(np.min(X[::2, :])-0.5, np.max(X[::2, :])+0.5)
plt.ylim(np.min(X[1::2,:])-0.5, np.max(X[1::2,:])+0.5)
colors = ['b']*N#plt.cm.jet(np.linspace(0, 1, N))

dot  = np.array([]) # for the planet

for c in colors:
    dot  = np.append(dot,  plt.plot([], [], 'o', c=c))
print(dot)
def animate(i):
    
    for k in range(N):
        
        dot[k].set_data(X[0, i, k], X[1, i, k])
    
    return dot

anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, T, 50), interval=1, blit=True, repeat=True)


plt.title('N body problem', fontsize=20)
plt.xlabel('X(t)', fontsize=20)
plt.ylabel('Y(t)', fontsize=20)

# Ucomment to save the animation, extra_args for .mp4
#anim.save('N_body.gif', fps=50)# extra_args=['-vcodec', 'libx264']) 

plt.show()
