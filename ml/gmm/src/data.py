import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


class Random2DGaussian:
    MIN_X1, MIN_X2 = -5, -5
    MAX_X1, MAX_X2 = 5, 5

    def __init__(self, mu: np.ndarray = None, sigma: np.ndarray = None):
        random_sample = np.random.random_sample

        self.mu = mu
        self.sigma = sigma

        if mu is None:
            self.mu = random_sample(size=(2,))

            self.mu[0] = self.MIN_X1 + (self.MAX_X1 - self.MIN_X1) * self.mu[0]
            self.mu[1] = self.MIN_X2 + (self.MAX_X2 - self.MIN_X2) * self.mu[1]

        if sigma is None:
            eigvalx1 = (random_sample() * (self.MAX_X1 - self.MIN_X1) / 5) ** 2
            eigvalx2 = (random_sample() * (self.MAX_X2 - self.MIN_X2) / 5) ** 2
            D = np.diag([eigvalx1, eigvalx2])

            phi = random_sample() * 2 * np.pi
            R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)],])

            self.sigma = R.T @ D @ R

        self.get_sample = lambda n: np.random.multivariate_normal(mu, sigma, n)


def sample_gauss_2d(C: int, N: int):
    gaussians = [Random2DGaussian() for _ in range(C)]

    X = np.vstack([gauss.get_sample(N) for gauss in gaussians])
    y = np.array(np.concatenate([[i] * N for i in range(C)]))

    return X, y


def plot_gaussian_2d(mean, cov, ax=None):
    """Plot 2D Gaussian distribution contour lines."""

    std_lvl5 = np.diag(6 * cov)

    x1 = np.linspace(mean[0] - std_lvl5[0], mean[0] + std_lvl5[0], num=200)
    x2 = np.linspace(mean[1] - std_lvl5[1], mean[1] + std_lvl5[1], num=200)

    X, Y = np.meshgrid(x1, x2)
    pos = np.dstack((X, Y))

    rv = multivariate_normal(mean, cov)

    ax = ax if ax is not None else plt
    ax.contour(X, Y, rv.pdf(pos), linewidths=1)


if __name__ == "__main__":
    gaussian1 = Random2DGaussian()
    gaussian2 = Random2DGaussian()

    plot_gaussian_2d(gaussian1.mu, gaussian1.sigma)
    plot_gaussian_2d(gaussian2.mu, gaussian2.sigma)

    plt.show()

