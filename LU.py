import numpy as np


# вектор Гаусса (вектор множителей). Функция возвращает вектор длины n - 1
def gauss(x):
    x = np.array(x, float)
    return x[1:] / x[0]


# A - исходная матрица, t - вектор Гаусса, функция выполняет А -> M1 * A
def gauss_app(A, t):
    A = np.array(A, float)
    t = np.array([[t[i]] for i in range(len(t))], float)
    A[1:, :] = A[1:, :] - t * A[0, :]
    return A


# определитель LU-матрицы
def determinant_in_lu(A):
    A = np.array(A, float)
    return A.diagonal().prod()


# LU-разложение матрицы
def lu_decomposition(A):
    LU = np.array(A, float)
    for k in range(LU.shape[0] - 1): # Lu.shape[0] возвращает количетсво строк в матриице
        max_ind = np.argmax(LU[k:, k]) + k
        max_elem = LU[max_ind, k]
        LU[[k, max_ind], k:] = LU[[max_ind, k], k:]
        # здесь создадим вектор перестановок p[k] = max_index
        if LU[k, k] != 0:
            t = gauss(LU[k:, k]) # вычисляем вектор множителей
            LU[k + 1:, k] = t # элементы с (k+1-го) до n в k-ом столбце матрицы L равны множителям Гаусса
            LU[k:, k + 1:] = gauss_app(LU[k:, k + 1:], t) # используем преобразование Гаусса
    return LU


# решение СЛАУ при помощи LU-разложения
def solve_lu(A, b):
    LU = lu_decomposition(A)
    b = np.array(b, float)
    for i in range(1, len(b)): # прямой ход
        b[i] = b[i] - np.dot(LU[i, :i], b[:i]) # метод .dot возвращает матричное перемножение
    for i in range(len(b) - 1, -1, -1): # обратный ход
        b[i] = (b[i] - np.dot(LU[i, i + 1:], b[i + 1:])) / LU[i, i]
    return b


# вычисление обратной матрицы
def inverse(A):
    E = np.eye(A.shape[0])
    Inv = []
    for e in E:
        x = solve_lu(A, e)
        Inv.append(x)
    return np.array(Inv).transpose()


# def test_lu_decomposition(A, LU)


def test_solve_lu():
    A = np.array([[1, 4, 7],
                  [2, 5, 8],
                  [3, 6, 10]])
    expected = np.array([-1./3, 1.3, 0])
    b = np.dot(A, expected)
    computed = solve_lu(A, b)
    tol = 1e-14
    success = np.linalg.norm(computed - expected) < tol
    msg = 'x_exact = ' + str(expected) + '; x_computed = '+ str(computed)
    if success:
        print(msg)
    else:
        print('Smth went wrong')


x = np.array([[1, 4, 7],
              [2, 5, 8],
              [3, 6, 10]], float)

# y = lu_decomposition(x)
# print(determinant_in_lu(y))
# print(np.linalg.det(x))

print(lu_decomposition(x))

