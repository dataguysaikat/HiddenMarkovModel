"""
_hmm_pure.py — Pure NumPy/SciPy Gaussian HMM.

Drop-in replacement for hmmlearn.hmm.GaussianHMM with a matching interface:
  .fit(X), .predict_proba(X), .transmat_, .means_, .covars_, .startprob_

Uses Baum-Welch EM with log-sum-exp scaling for numerical stability.
covariance_type="full" only (matches hmmlearn usage in hmm_model.py).
"""
from __future__ import annotations

import numpy as np
from scipy import linalg


def _log_multivariate_normal(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log-pdf of multivariate normal for each row of X. Shape: (T,)"""
    n = X.shape[1]
    diff = X - mean                          # (T, n)
    try:
        L = linalg.cholesky(cov, lower=True)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        sol = linalg.solve_triangular(L, diff.T, lower=True)  # (n, T)
        maha = np.sum(sol ** 2, axis=0)      # (T,)
    except linalg.LinAlgError:
        # Fallback: add small ridge if cov is singular
        reg = cov + np.eye(n) * 1e-6
        L = linalg.cholesky(reg, lower=True)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        sol = linalg.solve_triangular(L, diff.T, lower=True)
        maha = np.sum(sol ** 2, axis=0)
    return -0.5 * (n * np.log(2 * np.pi) + log_det + maha)


def _log_sum_exp(a: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log-sum-exp along axis."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis))
    return out + a_max.squeeze(axis=axis)


class GaussianHMM:
    """
    Gaussian Hidden Markov Model — pure NumPy/SciPy implementation.

    Parameters
    ----------
    n_components : int
    covariance_type : str  (only "full" supported)
    n_iter : int
    tol : float
    random_state : int | None
    """

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "full",
        n_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        self.startprob_: np.ndarray = np.ones(n_components) / n_components
        self.transmat_: np.ndarray = np.ones((n_components, n_components)) / n_components
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_params(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        T, n = X.shape
        K = self.n_components

        # K-means++ style initialisation for means
        idx = [rng.integers(T)]
        for _ in range(K - 1):
            dists = np.array([min(np.sum((X[i] - X[c]) ** 2) for c in idx) for i in range(T)])
            probs = dists / dists.sum()
            idx.append(rng.choice(T, p=probs))
        self.means_ = X[idx].copy().astype(float)

        # Covariances: global covariance + small identity
        global_cov = np.cov(X.T) + np.eye(n) * 1e-4
        self.covars_ = np.array([global_cov.copy() for _ in range(K)])

        # Transition matrix: slight self-persistence
        self.transmat_ = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(self.transmat_, 0.9)
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        self.startprob_ = np.ones(K) / K

    # ------------------------------------------------------------------
    # Log emission matrix
    # ------------------------------------------------------------------

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """Return (T, K) log-emission matrix."""
        T = X.shape[0]
        K = self.n_components
        log_B = np.empty((T, K))
        for k in range(K):
            log_B[:, k] = _log_multivariate_normal(X, self.means_[k], self.covars_[k])
        return log_B

    # ------------------------------------------------------------------
    # Forward / Backward (log-space)
    # ------------------------------------------------------------------

    def _forward(self, log_B: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Returns log_alpha (T, K) and log-likelihood.
        log_alpha[t, k] = log p(o_1..o_t, q_t=k)
        """
        T, K = log_B.shape
        log_A = np.log(self.transmat_ + 1e-300)
        log_pi = np.log(self.startprob_ + 1e-300)

        log_alpha = np.empty((T, K))
        log_alpha[0] = log_pi + log_B[0]

        for t in range(1, T):
            # (K,) = log-sum-exp over previous states
            log_alpha[t] = _log_sum_exp(log_alpha[t - 1, :, None] + log_A, axis=0) + log_B[t]

        log_likelihood = _log_sum_exp(log_alpha[-1], axis=0)
        return log_alpha, float(log_likelihood)

    def _backward(self, log_B: np.ndarray) -> np.ndarray:
        """Returns log_beta (T, K)."""
        T, K = log_B.shape
        log_A = np.log(self.transmat_ + 1e-300)

        log_beta = np.zeros((T, K))   # log_beta[T-1] = 0 (prob=1)
        for t in range(T - 2, -1, -1):
            log_beta[t] = _log_sum_exp(log_A + log_B[t + 1] + log_beta[t + 1], axis=1)

        return log_beta

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------

    def _compute_posteriors(
        self, log_alpha: np.ndarray, log_beta: np.ndarray, log_B: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        gamma[t, k]      = p(q_t=k | O, λ)               shape (T, K)
        xi[t, i, j]      = p(q_t=i, q_{t+1}=j | O, λ)   shape (T-1, K, K)
        """
        T, K = log_B.shape
        log_A = np.log(self.transmat_ + 1e-300)

        # gamma
        log_gamma = log_alpha + log_beta
        log_gamma -= _log_sum_exp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        # xi: (T-1, K, K)
        log_xi = (
            log_alpha[:-1, :, None]          # (T-1, K, 1)
            + log_A[None, :, :]              # (1,   K, K)
            + log_B[1:, None, :]             # (T-1, 1, K)
            + log_beta[1:, None, :]          # (T-1, 1, K)
        )
        log_xi -= _log_sum_exp(
            log_xi.reshape(T - 1, K * K), axis=1
        )[:, None, None]
        xi = np.exp(log_xi)

        return gamma, xi

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        K = self.n_components
        n = X.shape[1]
        eps = 1e-8

        self.startprob_ = gamma[0] + eps
        self.startprob_ /= self.startprob_.sum()

        # Transition matrix
        A_num = xi.sum(axis=0)             # (K, K)
        self.transmat_ = A_num / (A_num.sum(axis=1, keepdims=True) + eps)

        # Means and covariances
        gamma_sum = gamma.sum(axis=0) + eps   # (K,)
        self.means_ = (gamma[:, :, None] * X[:, None, :]).sum(axis=0) / gamma_sum[:, None]

        for k in range(K):
            diff = X - self.means_[k]        # (T, n)
            w = gamma[:, k]                  # (T,)
            cov = (w[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)
            self.covars_[k] = cov / gamma_sum[k] + np.eye(n) * 1e-4

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        self._init_params(X)
        prev_ll = -np.inf

        for _ in range(self.n_iter):
            log_B = self._log_emission(X)
            log_alpha, ll = self._forward(log_B)
            log_beta = self._backward(log_B)
            gamma, xi = self._compute_posteriors(log_alpha, log_beta, log_B)
            self._m_step(X, gamma, xi)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities, shape (T, K)."""
        log_B = self._log_emission(X)
        log_alpha, _ = self._forward(log_B)
        log_beta = self._backward(log_B)
        gamma, _ = self._compute_posteriors(log_alpha, log_beta, log_B)
        return gamma
