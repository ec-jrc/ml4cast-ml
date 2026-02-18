import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

n = 36
start_end = 0.1
peak = 0.2
center = (n - 1) / 2  # 17.5

x = np.arange(n)

a = (start_end - peak) / (center ** 2)
y = a * (x - center) ** 2 + peak
x =  x + 1

x = np.arange(n).reshape(-1, 1)

# fit
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(x)

# linear regression WITH intercept
model = LinearRegression(fit_intercept=True)
model.fit(X_poly, y)

# fitted values
y_fit = model.predict(X_poly)

print("Intercept:", model.intercept_)
for i, coef in enumerate(model.coef_, start=1):
    print(f"Coefficient x^{i}:", coef)

# plt.figure(figsize=(10,6))
# plt.scatter(x, y, c='black')
# plt.scatter(x, y_fit, c='red')
# plt.show()

plt.figure(figsize=(6, 4))
plt.plot(x.flatten(), y, "o-", color="black", label="Original")
plt.plot(x.flatten(), y_fit, "-", color="red", linewidth=2, label="Degree-3 fit")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()