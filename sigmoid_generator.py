
"""
forever:
    I give five parameters, the program uses it to plot sigmoid
    x = [1, ..., 30]
"""

from sigmoid_functions_v3 import SigmoidV3
import matplotlib.pyplot as plt

x_values = list(range(1, 31))

while True:
    input_string = input('Enter params: [a=* b=* c=* d=* f=*]\n')

    input_parts = input_string.split(' ')

    for i in range(len(input_parts)):
        input_parts[i] = float(input_parts[i].split('=')[1])

    series = [SigmoidV3.get_prediction(x, input_parts) for x in x_values]

    plt.plot(x_values, series)
    plt.show()
