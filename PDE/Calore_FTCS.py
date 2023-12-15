import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 100 #punti sulle x
x = np.linspace(0, N, N)
tstep = 5000 #punti sul tempo
T = np.zeros((N,tstep))

#Profilo di temperatura iniziale
T[0:N,0] = 500*np.exp(-((50-x)/20)**2)

D = 0.5
dx = 0.01
dt = 1e-4
r = D*dt/dx**2
#r < 1/2 affinche integri bene
print(r)

for time in range(1,tstep):
    for i in range(1,N-1):
        T[i,time] = T[i,time-1] + r*(T[i-1,time-1]+T[i+1,time-1]-2*T[i,time-1])

#    T[0,time] = T[1,time] #per avere bordi non fissi
#    T[N-1,time] = T[N-2,time]

fig = plt.figure(1)
ax = fig.gca(projection='3d')
gridx, gridy = np.meshgrid(range(tstep), range(N))
ax.plot_surface(gridx,gridy,T, cmap=mp.cm.coolwarm,vmax=250,linewidth=0,rstride=2, cstride=100)
ax.set_title('Diffusione del calore')
ax.set_xlabel('Tempo')
ax.set_ylabel('Lunghezza')
ax.set_zlabel('Temperatura')

fig = plt.figure(2)
plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(T), np.max(T))

line, = plt.plot([], [], 'b')
def animate(i):
    line.set_data(x, T[:,i])
    return line,


anim = animation.FuncAnimation(fig, animate, frames=tstep, interval=10, blit=True, repeat=True)

plt.grid()
plt.title('Diffusione del calore')
plt.xlabel('Distanza')
plt.ylabel('Temperatura')

#anim.save('calore.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()