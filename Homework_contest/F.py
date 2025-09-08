from typing import List, Self
from numbers import Number
from copy import deepcopy


class Matrix:
    def __init__(self, possible_matrix: List[List[Number]]) -> None:
        self.height = len(possible_matrix)
        self.length = len(possible_matrix[0]) if self.height > 0 else 0

        for i, row in enumerate(possible_matrix):
            if len(row) != self.length:
                raise ValueError(
                    f"Row {i} has length {len(row)}, expected {self.length}."
                )

        self.matrix = deepcopy(possible_matrix)

    def __str__(self) -> str:
        return "\n".join(" ".join(str(x) for x in row) for row in self.matrix)

    def __getitem__(self, key: tuple | slice):
        if isinstance(key, tuple):
            if all(isinstance(k, slice) for k in key):
                row_slice, col_slice = key
                new_matrix = [row[col_slice] for row in self.matrix[row_slice]]
                return Matrix(new_matrix)
            else:
                i, j = key
                return self.matrix[i][j]
        return self.matrix[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            i, j = key
            self.matrix[i][j] = value
        else:
            self.matrix[key] = value

    def __add__(self, other: Self) -> Self:
        if (self.height, self.length) != (other.height, other.length):
            raise ValueError("Matrix dimensions must match for addition.")

        return Matrix([
            [self[i, j] + other[i, j] for j in range(self.length)]
            for i in range(self.height)
        ])

    def __mul__(self, other: Self) -> Self:
        if (self.height, self.length) != (other.height, other.length):
            raise ValueError("Matrix dimensions must match for elementwise multiplication.")

        return Matrix([
            [self[i, j] * other[i, j] for j in range(self.length)]
            for i in range(self.height)
        ])

    def total(self) -> Number:
        return sum(sum(row) for row in self.matrix)


# Return a n x m Matrix of zeros
def zeroes(n: int, m: int) -> Matrix:
    return Matrix([[0] * m for _ in range(n)])


class Convolution(Matrix):
    pass


class Layer(Matrix):
    def apply(self, mask: Convolution) -> "Layer":
        if self.length < mask.length or self.height < mask.height:
            raise ValueError(
                f"Mask shape {mask.height}x{mask.length} must fit within Layer shape {self.height}x{self.length}."
            )

        new_layer = Layer(
            [[0] * (self.length - mask.length + 1) for _ in range(self.height - mask.height + 1)]
        )

        for i in range(self.height - mask.height + 1):
            for j in range(self.length - mask.length + 1):
                region = self[i:i+mask.height, j:j+mask.length]
                new_layer[i, j] = (region * mask).total()

        return new_layer


if __name__ == "__main__":
    n, m = map(int, input().split())
    convolution = Convolution([list(map(int, input().split())) for _ in range(n)])

    h, w = map(int, input().split())
    layer = Layer([list(map(int, input().split())) for _ in range(h)])

    new_layer = layer.apply(convolution)
    print(new_layer)
