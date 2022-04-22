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
    return A.diagonal().prod()


# LU-разложение матрицы
def lu_decomposition(A):
    LU = np.array(A, float)
    p = np.random.randint(-10, -1, size=A.shape[0]) # вектор перестановок
    for k in range(LU.shape[0]): # Lu.shape[0] возвращает количетсво строк в матриице
        max_ind = np.argmax(LU[k:, k]) + k # ищем индекс максимального элемента по строкам
        p[k] = max_ind  # вектор перестановок строк
        LU[[k, max_ind], :] = LU[[max_ind, k], :] # свапаем строки
        if LU[k, k] != 0:
            t = gauss(LU[k:, k]) # вычисляем вектор множителей
            LU[k + 1:, k] = t # элементы с (k+1-го) до n в k-ом столбце матрицы L равны множителям Гаусса
            LU[k:, k + 1:] = gauss_app(LU[k:, k + 1:], t) # используем преобразование Гаусса
    return (LU, p)


# решение СЛАУ при помощи LU-разложения
def solve_lu(A, b):
    res = lu_decomposition(A)
    LU = res[0]
    p = res[1] # вектор перестановок

    P = np.eye(LU.shape[0])  # вычисляем матрицу перестановок по вектору перестановок
    for i in range(p.shape[0]):
        P[[p[i], i], :] = P[[i, p[i]], :]

    b = np.array(np.dot(P, b), float) # b = Pb

    for i in range(1, len(b)): # прямой ход
        b[i] = b[i] - np.dot(LU[i, :i], b[:i]) # метод .dot возвращает матричное перемножение
    for i in range(len(b) - 1, -1, -1): # обратный ход
        b[i] = (b[i] - np.dot(LU[i, i + 1:], b[i + 1:])) / LU[i, i]
    return b


# вычисление обратной матрицы
def inverse_matrix(A):
    E = np.eye(A.shape[0])
    Inv = []
    for e in E:
        x = solve_lu(A, e)
        Inv.append(x)
    return np.array(Inv).transpose()


# проверка решения СЛАУ
def test_solve_lu():
    error = 1e-9

    # Test 1
    A1 = np.array([[1, 4, 7],
                  [2, 5, 8],
                  [3, 6, 10]])
    x1_expected = np.array([-1./3, 1.3, 0])
    b1 = np.dot(A1, x1_expected)
    x1_computed = solve_lu(A1, b1)
    statement = np.linalg.norm(x1_computed - x1_expected) < error
    print('\n\033[1;32;40m Test 1 is passed 👍👍👍 \n' if statement
          else f'\033[1;32;40m Test 1 is not passed ❌❌❌\nx_expected:\n{x1_expected}\nx_computed:\n{x1_computed}\n\n')

    # Test 2
    A2 = np.array([[1, -1, 3, 1],
                   [4, -1, 5, 4],
                   [2, -2, 4, 1],
                   [1, -1, 5, -1]])
    x2_expected = np.array([5, 4, 6, 3])
    b2 = np.dot(A2, x2_expected)
    x2_computed = solve_lu(A2, b2)
    statement = np.linalg.norm(x2_computed - x2_expected) < error
    print('\033[1;32;40m Test 2 is passed 👍👍👍 \n' if statement
          else f'\033[1;32;40m Test 2 is not passed ❌❌❌\nx_expected:\n{x2_expected}\nx_computed:\n{x2_computed}\n\n')

    # Test 3
    n = np.random.randint(2, 10)
    A3 = np.array([[22.75733654, 17.22065662, 22.57016195,  2.00718088, 6.14352123],
                  [14.33773534, 15.68986041, 10.72640792, 11.44631296, 17.26170242],
                  [19.77133746, 1.6417574, 16.02242907, 17.76892361, 12.03104429],
                  [2.58861734, 10.85832878, 12.16156742, 10.78991778,13.58864148],
                  [6.26408297, 10.52056427, 15.3906235, 3.59848876, 16.99920966]])

    b3 = np.array([-314.57460765, -184.08576518, -255.00163148, -115.43793868, -213.08198994])
    x3 = solve_lu(A3, b3)
    statement = np.linalg.norm(np.dot(A3, x3) - b3) < error # Ax - b = 0
    print('\033[1;32;40m Test 3 is passed 👍👍👍\n' if statement
          else f'\033[1;32;40m Test 3 is not passed ❌❌❌\n')


# проверка LU-разложения
def test_lu_decomposition():
    n = np.random.randint(2, 10)
    A = np.random.uniform(low=-10.5, high=23.1, size=(n, n))
    res = lu_decomposition(A)
    LU = res[0]
    p = res[1]

    print('A:', A, sep='\n')
    print('\n')

    L = np.tril(LU, -1) # нижнетреугольная матрица
    for i in range(L.shape[0]):
        L[i, i] = 1
    print('L:', L, sep='\n')
    print('\n')

    U = np.triu(LU) # верхнетреугольная матрица
    print('U:', U, sep='\n')

    P = np.eye(LU.shape[0]) # вычисляем матрицу перестановок по вектору перестановок
    for i in range(p.shape[0]):
        P[[p[i], i], :] = P[[i, p[i]], :]

    error = 1e-9
    statement = np.linalg.norm(np.dot(P, A) - np.dot(L, U)) < error # сравниваем PA и LU
    print('\n\033[1;32;40m LU is equal to PA 👍👍👍 \n' if statement
          else f'\033[1;32;40m LU is not equal to PA ❌❌❌\nLU:\n{np.dot(L, U)}\nPA\n{np.dot(P, A)}\n\n')


# проверка определителя
def test_determinant_in_lu():
    n = np.random.randint(2, 10)
    A = np.random.uniform(low=-10.5, high=23.1, size=(n, n))
    res = lu_decomposition(A)
    LU = res[0]
    p = res[1]

    print('A:', A, sep='\n')
    print('\n')

    print('LU:', LU, sep='\n')
    print('\n')

    error = 1e-4
    statement = np.abs(abs(determinant_in_lu(LU)) - abs(np.linalg.det(A))) < error

    print('\n\033[1;32;40m det(LU) is equal to det(A) 👍👍👍 \n' if statement
          else f'\033[1;32;40m LU is not equal to PA ❌❌❌\ndet(LU):\n{determinant_in_lu(LU)}\ndet(A):\n{np.linalg.det(A)}\n\n')


# проверка вычисления обратной матрицы
def test_inverse_matrix():
    error = 1e-9
    n = np.random.randint(2, 10)
    A = np.random.uniform(low=-10.5, high=23.1, size=(n, n))
    A_inversed = inverse_matrix(A)
    E = np.eye(n)

    # A * Ainv == Ainv * A == E
    statement = np.linalg.norm(np.dot(A, A_inversed) - E) < error and np.linalg.norm(np.dot(A_inversed, A) - E) < error

    print('\033[1;32;40m Inversed correclty 👍👍👍 \n' if statement
          else f'\033[1;32;40m Inversion failed ❌❌❌\nA * A_inversed:\n{np.dot(A, A_inversed)}\n')


test_lu_decomposition()
test_determinant_in_lu()
test_solve_lu()
test_inverse_matrix()
