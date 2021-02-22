import numpy as np

ws = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for w in ws[:-1]:
    print(w)
print("")
for i in range(len(ws)-1, 0, -1):
    print(ws[i-1])
