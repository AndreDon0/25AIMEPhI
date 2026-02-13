import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, (int, float, np.number)):
            data = np.array(data, dtype=np.float32)
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="add")

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad_self = out.grad

                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)

                for i, (gs, xs) in enumerate(zip(grad_self.shape, self.data.shape)):
                    if xs == 1 and gs != 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)

                self.grad += grad_self

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                grad_other = out.grad

                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)

                for i, (gs, xs) in enumerate(zip(grad_other.shape, other.data.shape)):
                    if xs == 1 and gs != 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)

                other.grad += grad_other

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other


    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="mul")

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad_self = out.grad * other.data

                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)

                for i, (gs, xs) in enumerate(zip(grad_self.shape, self.data.shape)):
                    if xs == 1 and gs != 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)

                self.grad += grad_self

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                grad_other = out.grad * self.data

                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)

                for i, (gs, xs) in enumerate(zip(grad_other.shape, other.data.shape)):
                    if xs == 1 and gs != 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)

                other.grad += grad_other

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op="sum")

        def _backward():
            if not self.requires_grad:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(self.data, 0)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _children=(self,), _op="relu")

        def _backward():
            if not self.requires_grad:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            grad = out.grad * (self.data > 0)
            self.grad += grad

        out._backward = _backward
        return out

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)
        for t in topo:
            t.grad = None


if __name__ == "__main__":
    np.random.seed(0)
    x = Tensor(np.random.randn(3, 2).astype(np.float32))
    W = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.zeros(4, dtype=np.float32), requires_grad=True)

    y = (x @ W + b).relu()
    loss = y.sum()
    loss.backward()

    print("loss:", loss.data)
    print("W.grad shape:", W.grad.shape)
    print("b.grad:", b.grad)