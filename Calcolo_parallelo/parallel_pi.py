import time
import random
import multiprocessing as mp

def compute_pi_s(N):
    inside = 0
    for _ in range(N):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1:
            inside += 1
    return 4 * inside / N


def compute_pi_p(N):
    inside = 0
    for _ in range(N):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1:
            inside += 1
    return inside

if __name__ == "__main__":

    N = int(1e8)

    start = time.time()
    pi    = compute_pi_s(N)
    end   = time.time() - start

    print(f"Pi greco: {pi}")
    print(f"Serial time: {end:.2f} secondi")

    # Stanzio tutti i processi che posso stanziare
    n_pro = mp.cpu_count()
    pool  = mp.Pool(processes=n_pro)

    # Ogni processo si divide equamente il numero di sample
    samples_per_process = N // n_pro

    start   = time.time()
    results = pool.map(compute_pi_p, [samples_per_process] * n_pro)
    inside  = sum(results)
    pi      = 4* inside / N
    end     = time.time() - start

    print(f"Pi greco: {pi}")
    print(f"Parallel time: {end:.2f} secondi")


