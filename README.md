# Домашнее задание к уроку 1: Основы PyTorch

## Задание 1: Создание и манипуляции с тензорами 

### 1.1 Создание тензоров 
```python
# Тензор размером 3x4, заполненный случайными числами от 0 до 1
random_tensor = torch.rand(3, 4)
# Тензор размером 2x3x4, заполненный нулями
zeros_tensor = torch.zeros(2, 3, 4)
# Тензор размером 5x5, заполненный единицами
ones_tensor = torch.ones(5, 5)
# Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
arange_tensor = torch.arange(16).reshape(4, 4)
```

### 1.2 Операции с тензорами

```python
# Дано: тензор A размером 3x4 и тензор B размером 4x3
a = torch.rand(3, 4)
b = torch.rand(4, 3)

# - Транспонирование тензора A
transposed_tensorA = a.T
# - Матричное умножение A и B
matrix_mul = a @ b
# - Поэлементное умножение A и транспонированного B
element_wise_mul = a * b.T
# - Вычислите сумму всех элементов тензора A
a_sum = a.sum()
```

### 1.3 Индексация и срезы

```python
# Тензор размером 5x5x5
tensor = torch.randint(1, 100, (5, 5, 5))

# Извлечение
# Первая строка
first_row = tensor[0, 0, :]
# Последний столбец
last_column = tensor[:, :, -1:]
# Подматрица размером 2x2 из центра тензора
center_2x2 = tensor[2, 1:3, 1:3]
# Все элементы с четными индексами
even_indices = tensor[:, :, ::2]
```


### 1.4 Работа с формами
```python
# Тензор размером 24 элемента
tensor = torch.arange(24)
# 2x12
tensor_2x12 = tensor.reshape(2, 12)
# 3x8
tensor_3x8 = tensor.reshape(3, 8)
# 4x6
tensor_4x6 = tensor.reshape(4, 6)
# 2x3x4
tensor_2x3x4 = tensor.reshape(2, 3, 4)
# 2x2x2x3
tensor_2x2x2x3 = tensor.reshape(2, 2, 2, 3)
```

## Задание 2: Автоматическое дифференцирование

### 2.1 Простые вычисления с градиентами
```python
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
```

#### Вывод 
```python
tensor(40.)
tensor(30.)
tensor(24.)
```

### 2.2 Градиент функции потерь

```python
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
```


### 2.3 Цепное правило

```python
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
x = torch.tensor(2.0, requires_grad=True)
f = torch.sin(x**2 + 1)
f.backward(retain_graph=True)

# Градиент df/dx
print("Градиент df/dx: ", x.grad.item())

# Результат с помощью torch.autograd.grad
autograd = torch.autograd.grad(f, x)[0]
print("Проверка через autograd.grad:", autograd.item())
```
#### Вывод 

```python
Градиент df/dx:  1.1346487998962402
Проверка через autograd.grad: 1.1346487998962402
```

