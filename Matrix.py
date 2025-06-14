import numpy as np
import time
import sys

def standard_mult(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if A[i][k] != 0:  # بهینه‌سازی جزئی
                for j in range(n):
                    C[i][j] += A[i][k] * B[k][j]
    return C

def strassen_mult(A, B):
    n = A.shape[0]
    if n <= 128:
        return standard_mult(A, B)
    
    mid = n // 2
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]
    
    M1 = strassen_mult(A11 + A22, B11 + B22)
    M2 = strassen_mult(A21 + A22, B11)
    M3 = strassen_mult(A11, B12 - B22)
    M4 = strassen_mult(A22, B21 - B11)
    M5 = strassen_mult(A11 + A12, B22)
    M6 = strassen_mult(A21 - A11, B11 + B12)
    M7 = strassen_mult(A12 - A22, B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    return np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

if __name__ == "__main__":
    sizes = [256, 512, 1024]
    results = {'Standard': [], 'Strassen': []}
    
    for n in sizes:
        print(f"Testing {n}x{n}...")
        A = np.random.randint(0, 10, (n, n))
        B = np.random.randint(0, 10, (n, n))
        
        start = time.time()
        standard_mult(A, B)
        std_time = time.time() - start
        
        start = time.time()
        strassen_mult(A, B)
        str_time = time.time() - start
        
        results['Standard'].append(std_time)
        results['Strassen'].append(str_time)
        
        print(f"Standard: {std_time:.2f}s, Strassen: {str_time:.2f}s")
    
    # Export results for plotting
    np.save('benchmark_results.npy', results)