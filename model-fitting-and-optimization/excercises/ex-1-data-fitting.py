import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
import os

# Generate data
x = np.arange(-12, 13, 1).reshape(-1, 1)
y = (x / 5) ** 3 + (x / 10) ** 2

noise = np.random.normal(0, 2, y.shape)
y_noise = y + noise

points = np.concatenate((x, y_noise), axis=1)

# Create subplots
fig, axs = plt.subplots(4, 2, figsize=(15, 20))
fig.suptitle('Data Fitting Techniques', fontsize=16)

# Original data and true function
for ax in axs.flat:
    ax.plot(x, y, label='True Function', color='blue')
    ax.scatter(x, y_noise, s=5, label='Noisy Data', color='orange')
    ax.grid()

# Delaunay Triangulation
axs[0, 0].triplot(points[:, 0], points[:, 1], Delaunay(points).simplices.copy())
axs[0, 0].set_title('Delaunay Triangulation')

# Linear Regression
model = LinearRegression()
model.fit(x, y_noise)
y_pred_linear = model.predict(x)
axs[0, 1].plot(x, y_pred_linear, color='red')
axs[0, 1].set_title('Linear Regression')

# Polynomial Regression
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)
model_poly = LinearRegression()
model_poly.fit(x_poly, y_noise)
y_pred_poly = model_poly.predict(x_poly)
axs[1, 0].plot(x, y_pred_poly, color='green')
axs[1, 0].set_title('Polynomial Regression (degree 3)')

# Spline Interpolation
spline = UnivariateSpline(x.flatten(), y_noise.flatten(), s=1)
y_spline = spline(x.flatten())
axs[1, 1].plot(x, y_spline, color='purple')
axs[1, 1].set_title('Spline Interpolation')

# LSQ Univariate Spline
knots = np.arange(-10, 11, 3)
lsq_spline = LSQUnivariateSpline(x.flatten(), y_noise.flatten(), knots)
y_lsq_spline = lsq_spline(x.flatten())
axs[2, 0].plot(x, y_lsq_spline, color='cyan')
axs[2, 0].set_title('LSQ Univariate Spline')

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(x, y_noise)
y_pred_ridge = ridge.predict(x)
axs[2, 1].plot(x, y_pred_ridge, color='brown')
axs[2, 1].set_title('Ridge Regression')

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(x, y_noise)
y_pred_lasso = lasso.predict(x)
axs[3, 0].plot(x, y_pred_lasso, color='pink')
axs[3, 0].set_title('Lasso Regression')

# Pull-Push Algorithm
def pull_push(x, y, levels=3):
    x_min, x_max = np.min(x), np.max(x)
    for level in range(levels):
        if len(y) % 2 == 1:
            y = np.append(y, y[-1])
        y = (y[:-1:2] + y[1::2]) / 2
        x = x[::2]

    for level in range(levels):
        y_upsampled = np.zeros(len(y) * 2 - 1)
        y_upsampled[::2] = y
        y_upsampled[1::2] = (y[:-1] + y[1:]) / 2
        y = y_upsampled
        x = np.linspace(x_min, x_max, len(y))

    return x, y

x_pull_push, y_pull_push = pull_push(x.flatten(), y_noise.flatten())
axs[3, 1].plot(x_pull_push, y_pull_push, color='magenta')
axs[3, 1].set_title('Pull-Push Algorithm')

# Add legends only to the first subplot for clarity
axs[0, 0].legend()

# Path
output_path = 'model-fitting-and-optimization/excercises/'

plt.savefig(os.path.join(output_path, 'output-data-fitting.png'))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


