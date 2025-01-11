# Neural Network Built From Scratch

## An implementation of a tiny neural network library featuring a custom 'Value' class that serves as a lightweight autograd engine. However, it's enough for building deep neural networks for solving the binary classification problem.

## Project structure
Project consists of the following components:
1. `neuralnet/` folder contains implementation of autograd engine and neural network:
    - `engine.py` - 'Value' class for representation of numeric values and computing gradients and backward passing.
    - `neuralnet.py` - implementation of 'Neuron', 'Layer' and 'MLP' classes for building multilayers perceptrons and working with them.

## Example of usage
Example of supported operations
```python
from engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).tanh()
d += 3 * d + (b - a).tanh()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 4.6252, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 27.0601, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 117.3345, i.e. the numerical value of dg/db
```

## Training a neural network
The notebook `demo.ipynb` provides a process of training a multilayer perceptron binary classifier. Used 2-layer neural net with 16 neurons each layer, MSE loss and stochastic gradient descent(SGD) for updating parameters. Classification result: 


![Classification result](reslut.jpg)


## License 
MIT License