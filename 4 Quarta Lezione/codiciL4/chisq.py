import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1, 2, 3])
y_data = np.array([1.5, 2.1, 3.3])
dy     = np.array([0.3]*y_data.size)

plt.figure(1, dpi=300)

plt.errorbar(x_data, y_data, dy, fmt='.', color='black', label='dati')

y1 = lambda x, m, q : x*m + q

t = np.linspace(np.min(x_data), np.max(x_data), 1000)
plt.xlim(0.7, 3.5)

#================== primo punto ==================================
plt.annotate("",
             xy=(0.95, 1.55), xycoords='data',
             xytext=(0.95, 1.15), textcoords='data',
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.annotate("d=1", xy=(0.75, 1.35))

plt.annotate("",
             xy=(1.05, 1.55), xycoords='data',
             xytext=(1.05, 0.95), textcoords='data',
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
             
plt.annotate("d$\simeq$ 1.7", xy=(1.1, 1.0))
#====================== secondo punto ==============================
plt.annotate("",
             xy=(2.05, y1(2, 1, 0.2)), xycoords='data',
             xytext=(2.05, 2.1), textcoords='data',
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.annotate(f"d $\simeq$ {(2.1-y1(2, 1, 0.2))/0.3:.1f}", xy=(2.1, 2.1))

plt.annotate("",
             xy=(1.95, 2.05), xycoords='data',
             xytext=(1.95, 2.55), textcoords='data',
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.annotate(f"d $\simeq$ {(2.1-y1(2, 1.5, -0.5))/0.3:.1f}", xy=(1.65, 2.55))
#====================== Terzo punto ==============================
plt.annotate("",
             xy=(3.05, y1(3, 1, 0.2)), xycoords='data',
             xytext=(3.05, 3.3), textcoords='data',
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.annotate(f"d $\simeq$ {(3.3-y1(3, 1, 0.2))/0.3:.1f}", xy=(3.1, 3.15))

plt.annotate("",
             xy=(2.95, 3.23), xycoords='data',
             xytext=(2.95, 4.05), textcoords='data',
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.annotate(f"d $\simeq$ {(3.3-y1(3, 1.5, -0.5))/0.3:.1f}", xy=(2.95, 3.8))

plt.plot(t, y1(t, 1, 0.2), 'b--', label=f'primo modello; $S^2$ = {sum(((y_data-y1(x_data, 1, 0.2))/0.3)**2):.1f}')
plt.plot(t, y1(t, 1.5, -0.5), 'r-.', label=f'secondo modello;  $S^2$ = {sum(((y_data-y1(x_data, 1.5, -0.5))/0.3)**2):.1f}')

plt.grid()
plt.legend(loc='best')
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("chisq_cfr.pdf")
plt.show()
