import math
import pandas as pd
import matplotlib.pylab as plt

data_combo = pd.read_excel('data.xlsx')
x1 = list(data_combo['x1'])
x2 = list(data_combo['x2'])
y = list(data_combo['y'])
yArr = []

def sigmoid(y):
    return 1 / (1 + math.exp(-y))

def updateWeight(w, d, x):
    return w + tetta * d * x

def updateBias(b, d):
    return b + tetta * d

# Boshlang'ich og'irliklar va boshqaruv qiymatlari
w1_1 = w1_2 = w1_3 = w1_4 = w1_5 = w1_6 = 0  # Kirishdan yashirin qatlamga og'irliklar
w2_1 = w2_2 = w2_3 = 0  # Elman qayta aloqa og'irliklari (yashirin qatlamdan)
w3_1 = w3_2 = w3_3 = 0  # Jordan qayta aloqa og'irliklari (chiqishdan)
w4_1 = w4_2 = 0  # Yashirin qatlamdan chiqishga og'irliklar
b1 = b2 = b3 = 0  # Boshqaruv qiymatlari (bias)
tetta = 0.01

# Boshlang'ich kontekstlar
hidden_context = [0, 0, 0]  # Elman uchun yashirin qatlam konteksti
output_context = 0  # Jordan uchun chiqish konteksti

for epoch in range(1000):
    loss = 0
    for t in range(len(x1)):
        x_1 = x1[t]
        x_2 = x2[t]
        y_act = y[t]

        # Yashirin qatlamni hisoblash: Elman (hidden_context) va Jordan (output_context) bilan
        z1 = (x_1 * w1_1 + x_2 * w1_2 +
              hidden_context[0] * w2_1 + output_context * w3_1 + b1)
        z2 = (x_1 * w1_3 + x_2 * w1_4 +
              hidden_context[1] * w2_2 + output_context * w3_2 + b2)
        z3 = (x_1 * w1_5 + x_2 * w1_6 +
              hidden_context[2] * w2_3 + output_context * w3_3 + b3)
        g1 = sigmoid(z1)
        g2 = sigmoid(z2)
        g3 = sigmoid(z3)

        # Chiqish qatlami
        z4 = g1 * w4_1 + g2 * w4_2 + b3
        g4 = z4  # Chiziqli chiqish
        y_pred = g4

        # Xatolarni hisoblash (backpropagation)
        d4 = y_act - y_pred
        d3 = w4_2 * d4 * g3 * (1 - g3)
        d2 = w4_1 * d4 * g2 * (1 - g2)
        d1 = w4_1 * d4 * g1 * (1 - g1)

        # Og'irliklar va boshqaruv qiymatlarini yangilash
        # Kirishdan yashirin qatlamga
        w1_1 = updateWeight(w1_1, d1, x_1)
        w1_2 = updateWeight(w1_2, d1, x_2)
        w1_3 = updateWeight(w1_3, d2, x_1)
        w1_4 = updateWeight(w1_4, d2, x_2)
        w1_5 = updateWeight(w1_5, d3, x_1)
        w1_6 = updateWeight(w1_6, d3, x_2)

        # Elman qayta aloqa og'irliklari (yashirin qatlamdan)
        w2_1 = updateWeight(w2_1, d1, hidden_context[0])
        w2_2 = updateWeight(w2_2, d2, hidden_context[1])
        w2_3 = updateWeight(w2_3, d3, hidden_context[2])

        # Jordan qayta aloqa og'irliklari (chiqishdan)
        w3_1 = updateWeight(w3_1, d1, output_context)
        w3_2 = updateWeight(w3_2, d2, output_context)
        w3_3 = updateWeight(w3_3, d3, output_context)

        # Yashirin qatlamdan chiqishga
        w4_1 = updateWeight(w4_1, d4, g1)
        w4_2 = updateWeight(w4_2, d4, g2)

        # Biaslarni yangilash
        b1 = updateBias(b1, d1)
        b2 = updateBias(b2, d2)
        b3 = updateBias(b3, d4)

        loss += d4 * d4

        # Kontekstlarni yangilash
        hidden_context = [g1, g2, g3]  # Elman uchun
        output_context = y_pred  # Jordan uchun

# Bashoratlarni hosil qilish
hidden_context = [0, 0, 0]  # Bashorat uchun kontekstlarni qayta boshlash
output_context = 0
for x_1, x_2 in zip(x1, x2):
    g1 = sigmoid(x_1 * w1_1 + x_2 * w1_2 +
                 hidden_context[0] * w2_1 + output_context * w3_1 + b1)
    g2 = sigmoid(x_1 * w1_3 + x_2 * w1_4 +
                 hidden_context[1] * w2_2 + output_context * w3_2 + b2)
    g3 = sigmoid(x_1 * w1_5 + x_2 * w1_6 +
                 hidden_context[2] * w2_3 + output_context * w3_3 + b3)

    g4 = g1 * w4_1 + g2 * w4_2 + b3
    y_pred = g4
    yArr.append(y_pred)

    # Kontekstlarni yangilash
    hidden_context = [g1, g2, g3]  # Elman uchun
    output_context = y_pred  # Jordan uchun

plt.plot(y, color='red')
plt.plot(yArr, color='blue')
plt.show()