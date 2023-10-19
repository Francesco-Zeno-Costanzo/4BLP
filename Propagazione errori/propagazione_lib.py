from uncertainties import ufloat
import uncertainties.umath as um

#il pimo argomento è il valore centrale, il secondo l'errore
x = ufloat(7.1, 0.2)
y = ufloat(12.3, 0.7)


print(x)
print(2*x-y)
print(um.log(x**y))
