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

def ubdateWeight(w, d, x):
    return w + tetta * d * x

def ubdateBias(b, d):
    return b + tetta * d

# Boshlang'ich og'irliklar va boshqaruv qiymatlari
w1_1 = w1_2 = w1_3 = w1_4 = w1_5 = w1_6 = 0  # Kirishdan yashirin qatlamga og'irliklar
w2_1 = w2_2 = w2_3 = w2_4 = w2_5 = w2_6 = 0  # Yashirin qatlamdan yashirin qatlamga (kontekst) og'irliklar
w3_1 = w3_2 = 0  # Yashirin qatlamdan chiqishga og'irliklar
b1 = b2 = b3 = b4 = b5 = b6 = 0  # Boshqaruv qiymatlari (bias)
tetta = 0.01

# Boshlang'ich kontekst (yashirin holat) birinchi qadam uchun
context = [0, 0, 0]  # Uchta yashirin tugun uchun boshlang'ich holat

for epoch in range(1000):
    loss = 0
    for t in range(len(x1)):
        x_1 = x1[t]
        x_2 = x2[t]
        y_act = y[t]

        # Kirishdan yashirin qatlamga hisoblash, kontekst bilan qayta aloqa
        z1 = x_1 * w1_1 + x_2 * w1_2 + context[0] * w2_1 + b1
        z2 = x_1 * w1_3 + x_2 * w1_4 + context[1] * w2_2 + b2
        z3 = x_1 * w1_5 + x_2 * w1_6 + context[2] * w2_3 + b3
        g1 = sigmoid(z1)
        g2 = sigmoid(z2)
        g3 = sigmoid(z3)

        # Ikkinchi yashirin qatlam, kontekst bilan qayta aloqa
        z4 = g1 * w2_4 + g2 * w2_5 + g3 * w2_6 + context[0] * w2_1 + b4
        z5 = g1 * w2_4 + g2 * w2_5 + g3 * w2_6 + context[1] * w2_2 + b5
        g4 = sigmoid(z4)
        g5 = sigmoid(z5)

        # Chiqish qatlami
        z6 = g4 * w3_1 + g5 * w3_2 + b6
        g6 = z6  # Chiziqli chiqish
        y_pred = g6

        # Xatolarni hisoblash
        d6 = y_act - y_pred
        d5 = w3_2 * d6 * g5 * (1 - g5)
        d4 = w3_1 * d6 * g4 * (1 - g4)
        d3 = (d4 * w2_6 + d5 * w2_6) * g3 * (1 - g3)
        d2 = (d4 * w2_5 + d5 * w2_5) * g2 * (1 - g2)
        d1 = (d4 * w2_4 + d5 * w2_4) * g1 * (1 - g1)

        # Og'irliklar va boshqaruv qiymatlarini yangilash
        w1_1 = ubdateWeight(w1_1, d1, x_1)
        w1_2 = ubdateWeight(w1_2, d1, x_2)
        w1_3 = ubdateWeight(w1_3, d2, x_1)
        w1_4 = ubdateWeight(w1_4, d2, x_2)
        w1_5 = ubdateWeight(w1_5, d3, x_1)
        w1_6 = ubdateWeight(w1_6, d3, x_2)

        w2_1 = ubdateWeight(w2_1, d4, context[0])
        w2_2 = ubdateWeight(w2_2, d4, context[1])
        w2_3 = ubdateWeight(w2_3, d4, context[2])
        w2_4 = ubdateWeight(w2_4, d5, g1)
        w2_5 = ubdateWeight(w2_5, d5, g2)
        w2_6 = ubdateWeight(w2_6, d5, g3)

        w3_1 = ubdateWeight(w3_1, d6, g4)
        w3_2 = ubdateWeight(w3_2, d6, g5)

        b1 = ubdateBias(b1, d1)
        b2 = ubdateBias(b2, d2)
        b3 = ubdateBias(b3, d3)
        b4 = ubdateBias(b4, d4)
        b5 = ubdateBias(b5, d5)
        b6 = ubdateBias(b6, d6)

        loss += d6 * d6

        # Keyingi qadam uchun kontekstni yangilash
        context = [g1, g2, g3]

# Bashoratlarni hosil qilish
for x_1, x_2 in zip(x1, x2):
    g1 = sigmoid(x_1 * w1_1 + x_2 * w1_2 + context[0] * w2_1 + b1)
    g2 = sigmoid(x_1 * w1_3 + x_2 * w1_4 + context[1] * w2_2 + b2)
    g3 = sigmoid(x_1 * w1_5 + x_2 * w1_6 + context[2] * w2_3 + b3)

    g4 = sigmoid(g1 * w2_4 + g2 * w2_5 + g3 * w2_6 + context[0] * w2_1 + b4)
    g5 = sigmoid(g1 * w2_4 + g2 * w2_5 + g3 * w2_6 + context[1] * w2_2 + b5)

    g6 = g4 * w3_1 + g5 * w3_2 + b6
    yArr.append(g6)

plt.plot(y, color='red')
plt.plot(yArr, color='blue')
plt.show()