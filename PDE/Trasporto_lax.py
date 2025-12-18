import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 100	    #numero punti sulle x
T = 400	    #numero di punti nel tempo
v = 1	    #velocità di propagazione
dt = 0.001	#passo temporale
dx = 0.01	#passo spaziale

alpha = v*dt/dx #<1
print(alpha)

Sol = np.zeros((N+1, T))
sol_v = np.zeros(N+1)
sol_n = np.zeros(N+1)

#condizione iniziale
q = 2*np.pi
x = np.linspace(0, (N+1)*dx, N+1)
sol_v = np.sin(q*1*x)
Sol[:, 0] = sol_v

#evoluzione temporale con lax
for time in range(1, T):
    for j in range(1, N):
        sol_n[j] = 0.5*(sol_v[j+1]*(1 - alpha)) + 0.5*(sol_v[j-1]*(1 + alpha))

    #condizione periodiche al bordo
    sol_n[0] = sol_n[N-1]
    sol_n[N] = sol_n[1]

    #aggiorno la soluzione
    sol_v = sol_n

    #conservo la soluzione per l'animazione
    Sol[:, time] = sol_v


fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.set_title('Equazione trasporto con Lax')
ax.set_ylabel('Distanza')
ax.set_xlabel('Tempo')
ax.set_zlabel('Ampiezza')


gridx, gridy = np.meshgrid(range(T), x)
ax.plot_surface(gridx, gridy, Sol)

plt.figure(2)
plt.title('Animazione soluzione', fontsize=15)
plt.xlabel('distanza')
plt.ylabel('ampiezza')
plt.grid()
plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(Sol[:,0]) - 0.1, np.max(Sol[:,0]) + 0.1)

line, = plt.plot([], [], 'b-')

def animate(i):

    line.set_data(x, Sol[:, i])
    return line,

anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, T, 1) ,interval=10, blit=True, repeat=True)

plt.figure(3)
plt.title('Soluzione a vari tempi', fontsize=15)
plt.xlabel('distanza')
plt.ylabel('ampiezza')
plt.grid()
col=plt.cm.jet(np.linspace(0, 1, 5))
for i, c in zip(np.arange(0, T, T//5), col):
    plt.plot(x, Sol[:, i], color=c, label=f'time={i*dt:.2f}')
plt.legend()
plt.show()
