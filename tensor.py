from numpy import argmax, argmin
import torch 
print(torch.__version__)

# Creating a Tensor

## using empty: allocates memory for a tensor
a = torch.empty(2, 3)
print(a)

## check type
print(type(a))

## using zeros
b = torch.zeros(2, 3)
## using ones
c = torch.ones(2, 3)

print(b)
print(c)

## using rand
d = torch.rand(2, 3)
print(d)

## using manual_seed
torch.manual_seed(100)
e = torch.rand(2, 3)
print(e)

## using torch.tensor
f = torch.tensor([[1, 2, 3], [4, 5, 6]])

## using arange: create tensor with a start, stop and step value
print("Using arange:", torch.arange(1, 10, 2))

## using linspace: create tensor with equally spaced values between start and stop
print("Using linspace: ", torch.linspace(0, 10, 10)) 

## using eye: create and identity tensor
print("Using eye:", torch.eye(5, 5))

## using full: fill a tensor with one particular value
print("Using full:", torch.full((3, 3), 5))
print()

# Reduction Operation

a = torch.randint(size = (2, 3), low = 0, high = 10, dtype=torch.float32)
print(a)

## sum
print(torch.sum(a))

## sum along columns
print(torch.sum(a, dim = 0))

## sum along rows
print(torch.sum(a, dim = 1))

## mean
print(torch.mean(a))

## mean along columns
print(torch.mean(a, dim = 0))

## mean along rows
print(torch.mean(a, dim = 1))

## median
print(torch.median(a))

## max and min: returns max and min value in the tensor
print(torch.max(a))
print(torch.min(a))

## product
print(torch.prod(a))

## standard deviation
print(torch.std(a))

## variance
print(torch.var(a))

## argmax and argmin: print the indices of the max and min value
print(argmax(a))
print(argmin(a))