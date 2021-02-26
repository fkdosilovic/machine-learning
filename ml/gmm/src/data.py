import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


class Random2DGaussian:
    MIN_X1, MIN_X2 = -5, -5
    MAX_X1, MAX_X2 = 5, 5

    def __init__(self, mean: np.ndarray = None, cov: np.ndarray = None):
        random_sample = np.random.random_sample

        self.mean = cov
        self.cov = cov

        if self.mean is None:
            self.mean = random_sample(size=(2,))

            self.mean[0] = self.MIN_X1 + (self.MAX_X1 - self.MIN_X1) * self.mean[0]
            self.mean[1] = self.MIN_X2 + (self.MAX_X2 - self.MIN_X2) * self.mean[1]

        if self.cov is None:
            eigvalx1 = (random_sample() * (self.MAX_X1 - self.MIN_X1) / 5) ** 2
            eigvalx2 = (random_sample() * (self.MAX_X2 - self.MIN_X2) / 5) ** 2
            D = np.diag([eigvalx1, eigvalx2])

            phi = random_sample() * 2 * np.pi
            R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)],])

            self.cov = R.T @ D @ R

        self.get_sample = lambda n: np.random.multivariate_normal(
            self.mean, self.cov, n
        )


def sample_gauss_2d(C: int, N: int):
    gaussians = [Random2DGaussian() for _ in range(C)]

    X = np.vstack([gauss.get_sample(N) for gauss in gaussians])
    y = np.array(np.concatenate([[i] * N for i in range(C)]))

    return X, y


def plot_gaussian_2d(mean, cov, ax=None):
    """Plot 2D Gaussian distribution contour lines."""

    std_lvl = 3 * np.sqrt(np.diag(cov))

    lower_bound = mean - std_lvl
    upper_bound = mean + std_lvl

    x1 = np.linspace(lower_bound[0], upper_bound[0], num=200)
    x2 = np.linspace(lower_bound[1], upper_bound[1], num=200)

    X, Y = np.meshgrid(x1, x2)
    pos = np.dstack((X, Y))

    rv = multivariate_normal(mean, cov)

    ax = ax if ax is not None else plt
    ax.contour(X, Y, rv.pdf(pos), linewidths=1)


def plot_data(X, y, y_pred=None, ax=None):
    """Plot the data in a feature space.

    Plots the data in a feature space. Correctly classified data is plotted as
    circles, while incorrectly classified data is plotted as x's.
    """
    colors = np.array(["b", "g", "r", "c", "m", "y", "k"])
    color = np.array([colors[label] for label in y])

    if y_pred == None:
        y_pred = y

    correct = y == y_pred
    incorrect = y != y_pred

    if ax is None:
        ax = plt

    ax.scatter(
        X[correct, 0], X[correct, 1], s=10, c=color[correct], marker="o",
    )

    ax.scatter(
        X[incorrect, 0], X[incorrect, 1], s=10, c=color[incorrect], marker="x",
    )


if __name__ == "__main__":
    gaussian1 = Random2DGaussian()
    gaussian2 = Random2DGaussian()

    plot_gaussian_2d(gaussian1.mu, gaussian1.sigma)
    plot_gaussian_2d(gaussian2.mu, gaussian2.sigma)

    plt.show()

