import matplotlib as mp
import matplotlib.pyplot as plt


#il file txt su cui scrivere se non esiste viene creato automaticamente

path_dati = "C:\\Users\\franc\\Desktop\\dati0.txt"
path_img = "C:\\Users\\franc\\Documents\\DatiL\\datiL3\\FIS2\\eOverm\\DSC_0005.jpg"

fig, ax = plt.subplots()

img = mp.image.imread(path_img)

ax.imshow(img)


def onclick(event):
    #apre file, il permesso e' a altrimenti sovrascriverebbe i dati
    file= open(path_dati, "a")

    x=event.xdata
    y=event.ydata
    print('x=%f, y=%f' %(x, y)) #stampa i dati sulla shell

    #scrive i dati sul file belli pronti per essere letti da codice del fit
    file.write(str(x))
    file.write('\t')
    file.write(str(y))
    file.write('\n')
    file.close() #chiude il file


fig.canvas.mpl_connect('button_press_event', onclick)


plt.show()
