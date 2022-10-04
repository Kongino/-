import matplotlib.pyplot as plt
import csv
import numpy as np


chul = []
pyo=open("C:/Users/inoh9/Desktop/ML/NoKada/afterNo.csv", 'r')
data = csv.reader(pyo)
i=1
for item in data:
    chul.insert(i, float(item[0]))
    i=i+1


print(chul)

plt.xlabel("Comb")
plt.ylabel("Accuracy")
plt.plot(chul)
plt.show()
