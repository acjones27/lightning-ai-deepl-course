# Unit 3

## Setup

```bash
cd unit2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

### Gradient descent
The loss measures “how wrong” the predictions are. And the gradient tells us how we have to change the weights to minimize (improve) the loss. The loss is correlated to the accuracy, but sadly, we cannot optimize the accuracy directly using stochastic gradient descent because accuracy is not a smooth function.

Computing the loss gradients is based on the chain rule from calculus. Pytorch can handle the differentiation automatically for us using automatic differentiation or autograd.

- Gradient descent updates the model parameters once per epoch
- Stochastic gradient descent (SGD) updates the model parameters after each example in the training data (so n updates per epoch where n is the number of training examples)
- Mini batch gradient descent (a flavour of stochastic gradient descent) updates the parameters once per mini batch (so m times where m < n). Typically m is some power of 2 e.g. 2^4, 2^5, 2^6 etc since this makes better use of GPU memory (TBD) and some concepts from linear algebra (TBD)

### autograd

According to the [pytorch docs](https://pytorch.org/docs/stable/autograd.html), `torch.autograd` provides classes and functions implementing automatic differentiation. You only need to declare Tensors for which gradients should be computed with the `requires_grad=True` keyword (currently only supported for floats). Then `backward` computes the sum of gradients of given tensors with respect to graph leaves, and `grad` computes and returns the sum of gradients of outputs with respect to the inputs. (TBD what all that means)

#### Example

Setup
```python
import torch

# parameters
w_1 = torch.tensor([0.23], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

# input and target
x_1 = torch.tensor([1.23])
y = torch.tensor([1.])

# computation graph
u = x_1 * w_1
z = u + b

# activation function
a = torch.sigmoid(z)

import torch.nn.functional as F
# Typical logistic regression loss
l = F.binary_cross_entropy(a, y)

# Better practice to use z directly instead of after the activation function
l = F.binary_cross_entropy_with_logits(z, y)
```

Grad
```python
from torch.autograd import grad

# retain graph so we can do the bias
grad_L_w1 = grad(l, w_1, retain_graph=True)

grad_L_b = grad(l, b)
```

### Other resources:

- [Blog on negative log-likelihood](https://sebastianraschka.com/blog/2022/losses-learned-part1.html)
- [Calculus and differentiation](https://sebastianraschka.com/pdf/books/dlb/appendix_d_calculus.pdf)
- [Alternative intro to SGD](https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html)
- [Course by Andrew Ng](https://www.coursera.org/learn/neural-networks-deep-learning/lecture/Z8j0R/binary-classification)
- [Notation guide for Andrew Ng course](https://d3c33hcgiwev3.cloudfront.net/_106ac679d8102f2bee614cc67e9e5212_deep-learning-notation.pdf?Expires=1722124800&Signature=XxJWVf8EVOrMbxfmRR2G-RpLXNRyPJXfaR6ZJNIgY~ySdeqmKm0COhvZsaMHCs3aMZ2rc4fOy7n6Jespm-7FsD-fsOmAwZ3wi2kYPzRP4nL1IcqNEodXIf4bkQ2ocmPrQs-GsmrT6ejTUZ4wYdkhMXRMElYLuXuxhSLNYnNM4VI_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)