def solve():
    L, d = map(int, input().split())

    A = []
    for _ in range(L):
        row = list(map(float, input().split()))
        A.append(row)

    C = []
    for _ in range(L):
        row = list(map(float, input().split()))
        C.append(row)

    V = []
    for _ in range(L):
        row = list(map(float, input().split()))
        V.append(row)

    for t in range(L):
        out_row = [0.0] * d

        A_t = A[t]

        for j in range(t + 1):
            C_j = C[j]
            V_j = V[j]

            dot_val = sum(A_t[k] * C_j[k] for k in range(d))

            for k in range(d):
                out_row[k] += dot_val * V_j[k]

        print(*(f"{x:.10f}" for x in out_row))

if __name__ == "__main__":
    solve()
