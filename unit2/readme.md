# Unit 2

## Setup

```bash
cd unit2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

### What are tensors?
- The main difference between tensors and arrays (e.g. torch.tensor vs numpy.array) is that tensors have GPU support and also support for automatic differentiation
- The main difference between tensors/arrays and native python lists is that tensors support many mathematical operations (e.g. mean) and the numerical computations are much faster
- A scalar is a rank-0 tensor, a vector is a rank-1 tensor, a matrix is a rank-2 tensor, etc

### Pytorch commands

Top 10 functions (see [slides](https://drive.google.com/file/d/1yFRPB9lnsq1QOzDknrNcEp9QgFSaTkRy/view); full API docs [here](https://pytorch.org/docs/stable/torch.html#tensors))
 
1. Creating tensors

```python
import torch
t = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
```

2. Checking the shape

```python
t.shape
# > torch.Size([2, 3])
```

3. Checking the rank

```python
t.ndim
# > 2
```

4. Checking the data type

```python

t.dtype
# > torch.float32
```

By default, floats in torch are 32 bit precision whereas integers are 64 bit. Note that a tensor can only have one type of data (unlike a python list)

5. Creating tensors from numpy arrays

We can do this in two ways: `torch.from_numpy(n)` or `torch.tensor(n)`. With `from_numpy`, PyTorch creates a tensor that shares the same memory as the NumPy array. In terms of memory usage, this is more efficient than `torch.tensor(n)`:

```python
import numpy as np
n = np.array([1., 2., 3.])
t = torch.from_numpy(n)
```

Note here that the dtype of these floats will be float64, since 64 bit is numpy's default precision for floats

6. Changing dtype

```python
t.to(torch.float32)
```

7. Checking the device type (i.e. where the tensor is in memory, either on CPU or GPU)

```python
t.device
# > device(type="cpu")
```

8. Change the tensor shape

```python
t = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

t2 = t.view(3, 2)

# Equivalent commands where python infers the missing dimension with placeholder -1 based on the number of elements in the tensor
t2 = t.view(-1, 2)
t2 = t.view(3, -1)
```

9. Transposing a tensor

```python
t.T
```

10. Multiplying matrices

```python
# Create a tensor of complementary shape for multiplying
t2 = t.T

t.matmul(t2)
```

Other resources:
- [Notebooks](https://github.com/Lightning-AI/dl-fundamentals/blob/main/unit02-pytorch-tensors/2.4-linalg/2.4-linalg-part1.ipynb) on linear algebra in pytorch
- [Pdb debugger](https://docs.python.org/3/library/pdb.html)