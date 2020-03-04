def f(x):
    return x ** 4 + 100

def df(x):
    return 4 * x ** 3

def ddf(x):
    return 12 * x ** 2

def parabaloid(x, y):
    return 4 * x ** 2 + y ** 2 + 29

def dp_dx(x, y):
    return 8 * x

def dp_dy(x, y):
    return 2 * y

def dp_ddx(x, y):
    return 8

def dp_ddy(x, y):
    return 2

def newton_optimize_parabola(x0, iterations):
    x = x0
    for i in range(iterations):
        x -= df(x) / ddf(x)

    return x, f(x)

def newton_optimize_parabaloid(x0, y0, iterations):
    x = x0
    y = y0

    for i in range(iterations):
        x -= dp_dx(x, y) / dp_ddx(x, y) * 1.999
        y -= dp_dy(x, y) / dp_ddy(x, y) * 1.999

    return x, y, parabaloid(x, y)

if __name__ == '__main__':
    print(newton_optimize_parabola(-10, 100))
    print(newton_optimize_parabaloid(-10, 10, 100000))
