'''
    Autograd is a core component of Pytorch that provides automatic differentiation for tensor operaitons. 
    It enables gradient computation.
'''

import torch

'''
Calculating dy/dx for y = x^2, x = 3

Return 2*x = 6
'''

# requires_grad determines whether gradients will be computed for that tensor during backpropagation. It is a boolean value, defaulting to False. When set to True, PyTorch tracks operations performed on the tensor, allowing for automatic differentiation.
x = torch.tensor(3.0, requires_grad=True)
print(x)

y = x**2
print(y)

# backward(): calculates gradients in backward direction
y.backward()
# .grad: prints the calculated value
print(x.grad)

'''
Calculating dz/dx for
z = sin(y)
y = x^2
x = 3

Return 2*x*cos(x^2) = -5.47
'''

x = torch.tensor(3.0, requires_grad=True)
y = x**2
z = torch.sin(y)

z.backward()
print(x.grad)

# Note:
# print(y.grad) throws an error as in the computation tree, 'y' is a non-leaf Tensor