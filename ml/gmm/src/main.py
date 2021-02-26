import data

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


class GaussianMixtureModel:
    def __init__(self, C: int, n_features: int):
        """Creates a new Gaussian Mixture Model."""
        self.C = C
        self.n_features = n_features

        self.means = np.random.uniform(low=-5, high=5, size=C * n_features).reshape(
            (C, n_features,)
        )
        self.covs = np.array([np.identity(n_features) for _ in range(C)])
        self.pi = np.array([1 / C] * C)

    def fit(self, X: np.ndarray):
        """Train GMM on the given input data, X, with EM algorithm."""
        nepochs = 50

        for _ in range(nepochs):
            # Create Gaussians based on means and covariance matrices.
            dists = [
                multivariate_normal(mean, cov)
                for mean, cov in zip(self.means, self.covs)
            ]

            # Evaluate responsibilities for every datapoint.
            R = np.vstack([dist.pdf(X) for dist in dists]).T
            R *= self.pi
            R /= np.sum(R, axis=1).reshape(-1, 1)

            # Calculate total responsability per class.
            Nks = np.sum(R, axis=0)

            # Update means.
            self.means = np.matmul(R.T, X) * (1 / Nks)

            # Update covariance matrices.
            for i, mean in enumerate(self.means):
                r_i = R[:, i].reshape(-1, 1, 1)
                cov_i = np.einsum("ij,ik->ijk", X - mean, X - mean)
                self.covs[i] = np.sum(r_i * cov_i, axis=0) / Nks[i]

                # print(np.linalg.det(self.covs[i]))

            # Update a posterior probabilities.
            self.pi = Nks / np.sum(Nks)

        print(self.covs)


if __name__ == "__main__":
    # np.random.seed(1337)

    # Sample the data.
    X, y = data.sample_gauss_2d(C=2, N=100)

    # Prepare the GMM.
    gmm = GaussianMixtureModel(C=2, n_features=2)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes = axes.flatten()

    # Plot the initial distributions.
    for mean, cov in zip(gmm.means, gmm.covs):
        data.plot_gaussian_2d(mean, cov, axes[0])

    gmm.fit(X.copy())

    # Plot the sampled data.
    data.plot_data(X, y, ax=axes[0])
    data.plot_data(X, y, ax=axes[1])

    # Plot the final distributions.
    for mean, cov in zip(gmm.means, gmm.covs):
        data.plot_gaussian_2d(mean, cov, axes[1])

    plt.show()
