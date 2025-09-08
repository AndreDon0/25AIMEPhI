def f(x):
    from math import sin
    return 2 * x - 15 * sin(x)

def iter(x_n, l, d_f):
    return x_n - l * d_f(x_n)

x_0, N, l = input().split()
x_0, N, l = float(x_0), int(N), float(l)

for _ in range(N):
    x_0 = iter(x_0, l, f)

res = f"{x_0:.4f}".rstrip('0')
print(res if res[-1] != '.' else res + '0')
