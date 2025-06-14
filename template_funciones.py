import numpy as np
import scipy.linalg
def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calculaLU(matriz):
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    A = matriz.copy()
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    for i in range(n-1):
        A = Ac.copy()
        factores = [0 for _ in range(i, n-1)] # Vector de factores multiplicativos para triangular
        pivot = A[i][i]

        for k in range(n-1-i): # Para k desde 1 hasta n-1 (en cada paso miro una fila menos, por eso desde i+1)
            factores[k] = - A[i+1+k][i] / pivot
        for f in range(i+1, n):
            factor_multiplicativo = factores[f-i-1]
            for c in range(i, n):
                if c == i:
                    Ac[f][c] = -factor_multiplicativo
                else:
                    Ac[f][c] = A[f][c] + A[i][c] * factor_multiplicativo
            
    L = np.tril(Ac,-1) + np.eye(A.shape[0]) 
    U = np.triu(Ac)
    
    return L, U

def transpuesta(matriz): # Recibe una matriz y devuelve su transpuesta.
    A = matriz.copy()
    n = matriz.shape[0]
    for f in range(n):
        for c in range(n):
            A[f][c] = matriz[c][f]
    return A

def existeSwapPosible(A, i): # Dada una matriz A y una columna i, devuelve el indice j > i de una fila f tal que A[f][i] != 0. Se usa dentro
# del calculo de la inversa para ver si puedo obtener un pivot distinto de 0 swapeando filas. Devuelve 1 si es posible, 0 sino.
    # Si no existe, devuelve -1
    n = A.shape[0]
    for f in range(n):
        if A[f][i] != 0:
            return f
    return -1

def swapRows(A, i, j): # Intercambia las filas i y j de la matriz A
    n = A.shape[0]
    for c in range(n):
        aux = A[i][c]
        A[i][c] = A[j][c]
        A[j][c] = aux
    return

def triangulacionInferior(A, I, Ac, n): # Subrutina de inversa. Triangula la matriz A por debajo de la diagonal. A es la matriz, I es la identidad a la que se le aplicarán los mismos cambios que a A para llevarla a la inversa, Ac es una copia de A, n es la dimensión de A (nxn). 
# No devuelve nada, trabaja por referencia.
     for i in range(n-1):
        A = Ac.copy()
        factores = [0 for _ in range(i, n-1)] # Vector de factores multiplicativos para triangular
        pivot = A[i][i]

        if(pivot == 0):
            toSwapIndex = existeSwapPosible(A, i)
            if(toSwapIndex == -1):
                return 'La matriz no es cuadrada (se anula alguna fila)'
            else:
                swapRows(A, i, toSwapIndex) # Intercambia las filas i con toSwapIndex 
            
        for k in range(n-1-i): # Para k desde 1 hasta n-1 (en cada paso miro una fila menos, por eso desde i+1)
            factores[k] = - A[i+1+k][i] / pivot
        for f in range(i+1, n):
            factor_multiplicativo = factores[f-i-1]
            for c in range(n):
                Ac[f][c] = A[f][c] + A[i][c] * factor_multiplicativo
                I[f][c] = I[f][c] + I[i][c] * factor_multiplicativo

def triangulacionSuperior(A, I, Ac, n): # Subrutina de inversa. 
# Triangula la matriz A por encima de la diagonal. Análoga a triangulacionSuperior. Trabaja por referencia con la matriz A, su copia Ac, su dimensión n y la identidad I (que en este punto ya no es la identidad sino la identidad modificada por triangulacionInferior).
    for i in range(n-1, 0, -1):
        A = Ac.copy()
        factores = [0 for _ in range(n-1)]
        pivot = A[i][i]
        for k in range(n-1):
            factores[k] = -A[k][i] / pivot
        for f in range(i):
            factor_multiplicativo = factores[f]
            for c in range(n):
                Ac[f][c] = A[f][c] + A[i][c] * factor_multiplicativo
                I[f][c] = I[f][c] + I[i][c] * factor_multiplicativo
    return

def inversa(A): # A matriz cuadrada inversible. Devuelve la inversa de A.
    n = A.shape[0]
    I= np.eye(n)
    Ac = A.copy().astype(float)

    # Triangulación debajo de la diagonal
    triangulacionInferior(A, I, Ac, n)
   
    triangulacionSuperior(A, I, Ac, n) # Triangula la matriz A por encima de la diagonal

    A = Ac.copy()
    # Multiplico por inversos multiplicativos de la diagonal las filas
    for f in range(n):
        inv_pivot = 1/A[f][f] 
        for c in range(n):
            A[f][c] = A[f][c] * inv_pivot
            I[f][c] = I[f][c] * inv_pivot

    return I


def calcula_matriz_K(A): # Recibe la matriz A de adyacencias y construye la matriz K como se define en el TP. Devuelve la matriz K.
    n = A.shape[0]
    K = np.zeros((n,n)) # Inicializa la matriz K
    for f in range(n):
        suma = 0
        for c in range(n):
            suma += A[f][c] # Suma de la fila i
        K[f][f] = suma # Coloca la suma en la diagonal de K
    return K


def calcula_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    K = calcula_matriz_K(A)
    Kinv = inversa(K) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = transpuesta(A)@Kinv # Calcula C multiplicando Kinv y A
    return C

    
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    k = 1-alfa
    I = np.eye(N)
    M = (N/alfa) * (I - k * C)

    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = np.ones((N, 1)) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N. d es alfa
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    n = D.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if(i==j):
                C[j][i] = 0
            else:
                numerador = 1/D[i][j]
                suma = 0
                for k in range(n):
                    if(k!=i):
                        suma = suma + 1/D[i][k]
                denominador = suma
                C[j][i] = numerador / denominador
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0]) # B = I
    r = cantidad_de_visitas
    C_k = C # C_k = C^1
    for k in range(1, r):
        # Sumamos las matrices de transición para cada cantidad de pasos
        B = B + C_k # B = B + C^k
        C_k = C_k @ C
    return B

# Funciones auxiliares del TP parte 1

def top3(vector):
    # Dado un vector de numeros reales positivos, devuelve un vector de tuplas (indice, valor) 
    # para los 3 de mayor valor del vector, ordenados de mayor a menor.
    #Obs: Se asume que el vector tiene más de 3 elementos
    copia = vector.copy()
    maximos = [-1, -1, -1]
    for j in range(3):
        indice_max = np.argmax(copia)
        copia[indice_max] = float('-inf')
        maximos[j] = indice_max
    return maximos


def norma1(A): # Función que calcula la norma 1 de una matriz (o sea la maxima sumatoria de una columna de la misma). Y la devuelve.
    max = 0
    for c in range(len(A)):
        col_sum = 0
        for f in range(len(A)):
            col_sum = col_sum + abs(A[f][c])
        if(col_sum > max):
            max = col_sum
    return max

def condB(B, r): # Calcula el número de condición de la matriz B con ||B|| = r conocido y lo devuelve. Usa r = ||B||_1 para reducir el error.
    # cond(B) = ||B|| * ||B^-1||
    # ||B|| = r
    B_inv = inversa(B)
    norma_inv = norma1(B_inv)
    return r * norma_inv


