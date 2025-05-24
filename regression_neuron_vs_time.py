import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1層ネットワークのデータ（analyze_results.pyより）
neuron_counts = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024]).reshape(-1, 1)
exec_times = np.array([73.41, 71.25, 76.73, 134.74, 185.02, 308.08, 624.75, 1205.1, 2521.6])

# 線形回帰
lin_reg = LinearRegression()
lin_reg.fit(neuron_counts, exec_times)
print(f"線形回帰式: 実行時間 = {lin_reg.coef_[0]:.3f} * ニューロン数 + {lin_reg.intercept_:.3f}")

# 2次多項式回帰
poly = PolynomialFeatures(degree=2)
neuron_counts_poly = poly.fit_transform(neuron_counts)
poly_reg = LinearRegression()
poly_reg.fit(neuron_counts_poly, exec_times)
print(f"2次回帰式: 実行時間 = {poly_reg.coef_[2]:.3f} * ニューロン数^2 + {poly_reg.coef_[1]:.3f} * ニューロン数 + {poly_reg.intercept_:.3f}")

# グラフ描画
plt.figure(figsize=(10, 6))
plt.scatter(neuron_counts, exec_times, color='blue', label='Measured')
plt.plot(neuron_counts, lin_reg.predict(neuron_counts), color='red', label='Linear Regression')
plt.plot(neuron_counts, poly_reg.predict(neuron_counts_poly), color='green', label='Quadratic Regression')
plt.xlabel('Total Neurons')
plt.ylabel('Execution Time [s]')
plt.title('Regression Analysis: Neuron Count vs Execution Time (Single Layer)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('regression_neuron_vs_time.png')
plt.show() 