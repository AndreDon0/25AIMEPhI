import numpy as np

class Tensor:
    # '_' mean - "Don't access or rely on these directly—treat as private, subject to change without notice."
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        # Почему 2 раза одно и то же? Как вообще преобразовать list(str)?
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
        self._op = _op  # Do we really need this?

    @property
    def shape(self):
        return self.data.shape

    # Fine
    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="add")

        def _backward():
            # 很么？See normal backward.
            if out.grad is None:
                return

            def compute_grad(tensor): # Зачем все это ксти в C все равно написано array[c*n + b]
                if tensor.grad is None:
                    tensor.grad = np.zeros_like(tensor.data)
                grad = out.grad

                while grad.ndim > tensor.data.ndim:
                    grad = grad.sum(axis=0)

                for i, (gs, xs) in enumerate(zip(grad.shape, tensor.data.shape)):
                    if xs == 1 and gs != 1:
                        grad = grad.sum(axis=i, keepdims=True)
                
                tensor.grad += grad

            if self.requires_grad:
                compute_grad(self)

            if other.requires_grad:
                compute_grad(other)

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
            
            def compute_grad(tensor):
                if tensor.grad is None:
                    tensor.grad = np.zeros_like(tensor.data)
                grad = out.grad * other.data

                while grad.ndim > tensor.data.ndim:
                    grad = grad.sum(axis=0)

                for i, (gs, xs) in enumerate(zip(grad.shape, tensor.data.shape)):
                    if xs == 1 and gs != 1:
                        grad = grad.sum(axis=i, keepdims=True)

                tensor.grad += grad

            if self.requires_grad:
                compute_grad(self)

            if other.requires_grad:
                compute_grad(other)

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

    def matmul(self, other): # В torch такое было?
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad @ other.data.T  # Математика, которую я не изучал, а вот Qwen...
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other): # Didn't know this.
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False): # Думаю keepdim лишний...
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

    # WOW!!! torch.tensor([-1, 0, 1]).relu() actually exist!
    def relu(self):
        out_data = np.maximum(self.data, 0)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _children=(self,), _op="relu")

        def _backward():
            if not self.requires_grad:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            grad = out.grad * (self.data > 0) # That's why I love relu. It's compute so fast!
            self.grad += grad

        out._backward = _backward
        return out

    # 很好。
    @staticmethod
    def build_topo(t, _visited=None):
        if _visited is None:
            _visited = set()
        topo = []
        if t not in _visited:
            _visited.add(t)
            for child in t._prev:
                topo.extend(Tensor.build_topo(child, _visited=_visited))
            topo.append(t)
        return topo

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) # 没错。

        topo = Tensor.build_topo(self)

        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        topo = Tensor.build_topo(self)
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