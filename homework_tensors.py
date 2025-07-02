import torch


# 1.1 Создание тензоров

# Тензор размером 3x4, заполненный случайными числами от 0 до 1
random_tensor = torch.rand(3, 4)
# Тензор размером 2x3x4, заполненный нулями
zeros_tensor = torch.zeros(2, 3, 4)
# Тензор размером 5x5, заполненный единицами
ones_tensor = torch.ones(5, 5)
# Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
arange_tensor = torch.arange(16).reshape(4, 4)


# 1.2 Операции с тензорами

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


# 1.3 Индексация и срезы

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



# 1.4 Работа с формами

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
