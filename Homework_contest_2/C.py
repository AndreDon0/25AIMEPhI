import numpy as np

A = np.array(list(map(int, input().split())))
B = np.array(list(map(int, input().split())))

if len(A) != len(B):
    print("No solution")
else:
    print(f"{np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)):.2}")