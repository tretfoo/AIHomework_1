import torch


# 2.1 Простые вычисления с градиентами

# Тензоры x, y, z с requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(6.0, requires_grad=True)

func = x**2 + y**2 + z**2 + 2 * x * y * z

# Нахожение градиентов по всем переменным
func.backward()
x_grad = x.grad
y_grad = y.grad
z_grad = z.grad

print(x_grad)  # 2x + 2yz = 4 + 36 = 40
print(y_grad)  # 2y + 2xz = 6 + 24 = 30
print(z_grad)  # 2z + 2xy = 12 + 12 = 24


# 2.2 Градиент функции потерь

# Функция MSE
def MSE(x, w, b, y_true):
    """Mean Squared Error 

    Args:
        x (torch.Tensor): входные данные
        w (torch.Tensor): вес модели
        b (torch.Tensor): смещение
        y_true (torch.Tensor): реальные значения

    Returns:
        torch.Tensor: Cреднеквадратичная ошибка
    """
    y_pred = w * x + b
    mse = torch.mean((y_pred - y_true) ** 2)
    return mse

loss = MSE(x, y_true, w, b)

# Градиенты по w и b
loss.backward()
w_grad = w.grad
b_grad = b.grad


# 2.3 Цепное правило

# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
x = torch.tensor(2.0, requires_grad=True)
f = torch.sin(x**2 + 1)
f.backward(retain_graph=True)

# Градиент df/dx
print("Градиент df/dx: ", x.grad.item())

# Результат с помощью torch.autograd.grad
autograd = torch.autograd.grad(f, x)[0]
print("Проверка через autograd.grad:", autograd.item())


