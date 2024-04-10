"""
Codice per risolvere il problema delle N regine in una scacchiera N x N
"""
import matplotlib.pyplot as plt


def queens(n, i=0, col=[], diag_sup=[], diag_inf=[]):
    '''
    Codice che genera le configurazioni delle N regine,
    la funzione è un generatore quindi non restituisce
    tuttle le configurazioni ma le calcola man mano

    Parameters
    ----------
    n : int
        quante regine vanno piazzate
    i : int
        numero di regine piazzate
    col : list
        lista che contiene la posizione della regina
        nelle varie colonne.
    diag_sup : list
        lista per controllare le diagonali in salita
    diag_inf : list
        lista per controllare le diagonali in discesa
    '''

    # Finchè non sono piazzate N regine
    if i < n:
        # Ciclo sulle posizioni
        for j in range(n):
            # se la regina non è nella colnna e non ci stanno delle regine
            # a minacciare la casella lungo le diagonali, chiamo ricorsivamente
            if j not in col and i + j not in diag_sup and i - j not in diag_inf:
                # richiamo avendo fissato una regina nella posizione j
                yield from queens(n, i + 1, col + [j], diag_sup + [i + j], diag_inf + [i - j])
    else:
        yield col


def plot_queens(board):
    '''
    Funzione per plottare la scacchiera

    Parameter
    ---------
    board : list
        lista delle posizioni, ouput della funzione queens
    '''

    n = len(board)

    # Creo la scacchiera
    chessboard = [[(i + j) % 2 for i in range(n)] for j in range(n)]
    plt.imshow(chessboard, cmap='binary')

    # Metto le regine
    for i in range(n):
        plt.text(i, board[i], 'Q', color='red', ha='center', va='center', fontsize=20)

    plt.yticks(range(n), [i for i in range(n, 0, -1)])
    plt.xticks(range(n), [chr(i) for i in range(97, 97+n)])
    plt.savefig('8Q.pdf')
    plt.show()


def select(N):
    '''
    funzione per selezionare l'N-esima configurazione
    '''
    Q = queens(8)
    for i in range(N-1):
        next(Q)

    return next(Q)

B = select(11)
plot_queens(B)

