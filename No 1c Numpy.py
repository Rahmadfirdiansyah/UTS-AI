import numpy as np
inputs = [[1.2, 0.8, 0.5, 0.2, 0.1, 0.4],
					 [0.5, 0.91, 0.26, 0.5, 0.13, 0.15],
					 [0.26, 0.27, 0.17, 0.87, 0.77, 0.11],
                     [0.31, 0.33, 0.34, 0.35, 0.37, 0.38],
                     [0.21, 0.22, 0.23, 0.24, 0.25, 0.26],
                     [0.2, 0.8, 0.5, 0.2, 0.1, 0.4],
					 [0.5, 0.91, 0.26, 0.5, 0.13, 0.15],
					 [0.26, 0.27, 0.17, 0.87, 0.77, 0.11],
                     [0.31, 0.33, 0.34, 0.35, 0.37, 0.38],
                     [0.21, 0.22, 0.23, 0.24, 0.25, 0.26]]
                     
weights = [[0.2, 0.8, 0.5, 0.2, 0.1, 0.4],
					 [0.5, 0.91, 0.26, 0.5, 0.13, 0.15],
					 [0.26, 0.27, 0.17, 0.87, 0.77, 0.11],
                     [0.31, 0.33, 0.34, 0.35, 0.37, 0.38],
                     [0.21, 0.22, 0.23, 0.24, 0.25, 0.26]]
                     
biases = [2.0, 3.0, 0.5, 1.5, 2.0]
layer_outputs = np.dot(inputs, np.array(weights).T) + biases 
print(layer_outputs)