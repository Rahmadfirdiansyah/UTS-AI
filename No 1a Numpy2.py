#menggunakan fungsi numpy
import numpy as np

#input data
inputs = [  [1.3, 3.3, 3.4, 5.5, 5.7, 7.9, 1.4, 3.9, 3.5, 4.9],
            [0.5, 0.5, 1.3, 3.5, 3.5, 4.5, 1.5, 3.9, 5.9, 5.5],
            [0.3, 7.5, 5.5, 9.7, 4.5, 5.7, 4.5, 0.9, 7.7, 5.5],
            [1.1, 3.3, 4.1, 5.1, 5.3, 5.3, 5.9, 9.1, 4.3, 5.5],
            [3.3, 4.4, 5.3, 1.5, 0.3, 0.5, 5.3, 7.3, 5.3, 3.1],
            [4.7, 0.5, 0.4, 0.7, 0.1, 0.3, 5.3, 5.9, 5.9, 5.7]]

#panjang weights
weights1 = [[1.3, 3.3, 3.4, 4.5, 5.5, 5.5, 9.5, 5.5, 9.1, 7.3],
            [5.7, 7.9, 9.1, 3.4, 3.5, 3.5, 3.3, 1.5, 3.5, 5.5],
            [5.3, 5.3, 4.3, 1.9, 9.7, 7.5, 1.4, 3.1, 5.5, 5.5],
            [5.5, 5.5, 5.4, 4.3, 3.3, 3.1, 5.5, 5.1, 3.3, 1.1],
            [7.4, 5.4, 5.9, 5.1, 9.3, 7.3, 3.5, 5.3, 3.1, 0.3]]

#biases pada layer1
biases1 = [4.4, 1.3, 3.5, 4.5, 5.9]

#variable 3
weights3 =  [   [0.1, 3.3, 3.4, 5.4, 4.5],
                [4.3, 1.1, 3.5, 5.7, 5.5],
                [3.5, 3.4, 9.7, 7.5, 5.4]]

#biases pada layer3
biases3= [3.4, 5.1, 5.5]

#menghitung layer1 menggunakan inputs, weights1, dan biases 1 menggunakan perkalian dot
layer1 = np.dot(inputs, np.array(weights1).T) + biases1

#menghitung layer3 dari hasil perhitungan layer1 menggunakan dot
layer3 = np.dot(layer1, np.array(weights3).T) + biases3

#print output layer3
print(layer3)