import numpy as np

H, W, N, T, K = map(int, input().split())

M = np.ones((H + 2, W + 2), dtype=int)
M[1:-1, 1:-1] -= np.ones((H, W), dtype=int)
for _ in range(N):
    h, w = list(map(int, input().split()))
    M[h + 1, w + 1] = 1

A = list([input() for _ in range(T)])

print(M)

p = [1, 1] # y x
code = {'U': 0, 'R': 1, 'D': 2,'L': 3}
anticode = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
action = {}
for t in range(1, T):
    py, px = p
    su, sr, sd, sl = M[py-1, px], M[py, px+1], M[py, px], M[py, px] # u r d l
    a = code[A[t - 1]]
    action[int(a + o[0]*2**2 + o[1]*2**3 + o[2]*2**4 + o[3]*2**5)] = a
    if a == 0:
        p[0] -= 1
    elif a == 1:
        p[1] += 1
    elif a == 2:
        p[0] += 1
    elif a == 3:
        p[1] -= 1

print(action)

for t in range(T, T + K):
    o = [M[p[0]-1, p[1]], M[p[0], p[1]+1], M[p[0]+1, p[1]], M[p[0]+1, p[1]]]
    s = action[int(a + o[0]*2**2 + o[1]*2**3 + o[2]*2**4 + o[3]*2**5)]
    print(anticode[s])
    a = s