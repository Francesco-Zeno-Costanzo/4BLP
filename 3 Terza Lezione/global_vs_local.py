import timeit
import dis

start = timeit.default_timer()

for i in range(int(1e8)):
    pass

end = timeit.default_timer() - start

print(f"Tempo in gloable = {end}")

def f():
    for i in range(int(1e8)):
        pass

start = timeit.default_timer()
f()
end = timeit.default_timer() - start

print(f"Tempo in locale  = {end}")

print(dis.dis(f))
