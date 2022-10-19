#menggunakan fungsi numpy
import numpy as np

#input data
inputs = [1.6, 1.5, 2.0, 2.4, 1.5, 1.0, 4.0, 3.5, 1.3, 2.0]
weights = [[0.2, 0.9, 0.5, 0.1, 0.2, 0.3, 0.5, 0.5, 0.6, 0.7],]
biases = [2.0, 2.0, 0.5, 1.5, 2.4]
#perkalian Weight dan inputs menggunakan dot lalu ditambah oleh biases
hasil = np.dot(weights, inputs) + biases
#print output
print(hasil)