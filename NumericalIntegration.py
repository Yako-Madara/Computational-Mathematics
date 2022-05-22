import numpy as np
import time 

# Метод ячеек для вычисления двойного интеграла
class CellMethod:
    def __init__(self, eps = 0.0001, a = 0, b = 1) -> None:
        """
        Args:
            eps (float, optional): заданная точность. Defaults to 0.0001.
            a (int, optional): нижняя граница. Defaults to 0.
            b (int, optional): верхняя граница. Defaults to 1.
        Для u и v границы одиннаковые, так как переходим к квадратной области определения.
        """
        self.eps = eps
        self.a = a
        self.b = b
        self.k = 2
        self.report = [] # переменная для сохранения параметров интегрирования
         
    # Преобразованная интегрируемая функция в явном виде
    def F(self, u,v):
        return u * (u**2 + (1 + u - u**2) * v)**2 * (1 + u - u**2)
    
    def integrate(self, h):
        I = 0
        # Сгенерируем сетку 
        x = np.arange(0,1+h,h)
        y = np.arange(0,1+h,h)
        # Площадь ячейки
        S = h**2
        # Считаем интеграл
        for i in range(1, len(x)):
            for j in range(1, len(y)):
                I += S * self.F((x[i-1] + x[i]) / 2, (y[j-1] + y[j]) / 2)
        return I 
    
    def solve(self):
        I_true = 0.775
        # Создаем начальную сетку
        # Количество узлов
        n = int((self.b - self.a) / np.sqrt(self.eps)) + 1
        # начальный шаг сетки
        h = (self.b - self.a) / n
        # Найдем начальное значение интеграла с заданной сеткой
        t1 = time.time()
        I_h = self.integrate(h)
        t2 = time.time()
        delta = np.abs(I_true-I_h) / I_true
        self.report.append((h, I_h, delta, float(t2-t1))) 
        # Считаем интеграл с заданной точностью по правилу Рунге
        while True:
            t1 = time.time()
            I_h_2 = self.integrate(h/2)
            t2 = time.time()
            delta = np.abs(I_true-I_h_2) / I_true
            self.report.append((h/2, I_h_2, delta, float(t2-t1))) 
            if np.abs(I_h_2 - I_h) / 3 < self.eps:
                I_h = I_h_2
                break
            else:
                h = h / 2 
                I_h = I_h_2      
        #return I_h 
    
# Метод Монте-Карло для вычисления двойного интеграла
class MonteCarlo:
    def __init__(self, f = None, fi1=None, fi2=None, a=None, b=None) -> None:
        """
        Args:
            f (_type_): интегрируемая функция
            fi1 (_type_): нижняя граница для y (функция)
            fi2 (_type_): верхняя граница для y (функция)
            a (_type_): нижняя граница для x
            b (_type_): верхняя граница для x
        """
        self.f = f
        self.fi1 = fi1
        self.fi2 = fi2
        self.a = a
        self.b = b
    
    def solve(self, n:int = 10):
        """Вычисление двойного интеграла

        Args:
            n (int): количество точек
        """
        I = None # Значение интеграла
        # генерируем точки 
        u = np.random.sample(n)
        v = np.random.sample(n)
        # новый вид функции F(u,v), записанный явно
        def F(u,v):
            return u * (u**2 + (1 + u - u**2) * v)**2 * (1 + u - u**2)
        I = np.mean(F(u,v))
        return I
        
""" 
Тестовая функция и ее область определения
a = 0 
b = 1
# Функция фи1(x) 
def fi_1(x):
    return x**2
# Функция фи2(x)
def fi_2(x):
    return 1 + x
# Интегрируемая тестовая функция f(x,y)
def f(x,y):
    return x * y**2 
"""

# Вычислительный эксперимент
if __name__ == '__main__':
    I_true = 0.775
    np.random.seed(10)
    
    test_m_k = MonteCarlo()
    N = [10,100,1000,10000,100000, 1000000, 10000000]
    for n in N: 
        t1 = time.time()
        I = test_m_k.solve(n)
        t2 = time.time()
        t = float(t2-t1)
        delta = np.abs(I_true-I) / I_true
        print('n=%d, I=%.5f, delta=%.5f, time=%.3f' % (n, I, delta, t)) 
    
    print('\n\n')
    
    test_c_m = CellMethod()
    t1 = time.time()
    test_c_m.solve()
    t2 = time.time()
    t_sum = float(t2-t1)
    for i in range(len(test_c_m.report)):
        frame = test_c_m.report[i]
        print('h=%.6f, I=%.5f, delta=%.5f, time=%.3f' % (frame[0], frame[1], frame[2], frame[3]))
    print('Суммарное время: %.3f' % t_sum)