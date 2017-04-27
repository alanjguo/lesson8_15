import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate output error
error = target - output
print ('Error: ',error)

# TODO: Calculate error term for output layer
output_error_term = error * output * (1 - output)

# TODO: Calculate error term for hidden layer

# This works because in nparray, [-1,2]*[1,2]=[-1,4]
hidden_error_term = np.dot(output_error_term, weights_hidden_output) * \
                    hidden_layer_output * (1 - hidden_layer_output)

print ('Shapes: output_error_term', output_error_term)
print ('weights_hidden_output:', weights_hidden_output)
print ('Shapes: hidden_layer_output', hidden_layer_output)
print ('Shapes dot product: ', np.dot(output_error_term, weights_hidden_output) )
print ('hidden_error_term: ', hidden_error_term)
print ('Hiden Layer Output ', hidden_layer_output)
print (np.dot(output_error_term, weights_hidden_output))


# TODO: Calculate change in weights for hidden layer to output layer
# This is dimension 1*2
delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
# This is dimension 3*2 = c * [1*2] * [3*1]
delta_w_i_h = learnrate * hidden_error_term * x[:, None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
