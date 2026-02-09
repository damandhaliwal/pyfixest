import warnings
from math import sqrt
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

from pyfixest.estimation.cupy.demean_cupy_ import create_fe_sparse_matrix


def _convert_scipy_sparse_to_torch(
    scipy_csr: Any, dtype: "torch.dtype", device: "torch.device"
) -> "torch.Tensor":
    """Convert scipy CSR sparse matrix to PyTorch sparse CSR tensor."""
    scipy_csr = scipy_csr.tocsr()
    crow_indices = torch.from_numpy(scipy_csr.indptr.astype(np.int64)).to(device)
    col_indices = torch.from_numpy(scipy_csr.indices.astype(np.int64)).to(device)
    values = torch.from_numpy(scipy_csr.data.copy()).to(dtype=dtype, device=device)
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, size=scipy_csr.shape, dtype=dtype, device=device
    )


def lsmr_pytorch(
    A: "torch.Tensor",
    b: "torch.Tensor",
    damp: float = 0.0,
    atol: float = 1e-8,
    btol: float = 1e-8,
    maxiter: Optional[int] = None,
) -> tuple["torch.Tensor", int, int]:
    """
    LSMR algorithm for sparse least-squares problems, implemented in PyTorch.

    Solves min ||Ax - b||_2 + damp * ||x||_2 using the Golub-Kahan
    bidiagonalization process following Fong & Saunders (2011).

    Parameters
    ----------
    A : torch.Tensor
        Sparse CSR matrix of shape (m, n).
    b : torch.Tensor
        Right-hand side vector of shape (m,).
    damp : float, default=0.0
        Damping parameter (Tikhonov regularization).
    atol : float, default=1e-8
        Absolute tolerance for convergence.
    btol : float, default=1e-8
        Relative tolerance for convergence.
    maxiter : int, optional
        Maximum number of iterations. Defaults to min(m, n).

    Returns
    -------
    x : torch.Tensor
        Solution vector of shape (n,).
    istop : int
        Convergence code:
        0 = initial x = 0 is exact solution,
        1 = ||Ax - b|| sufficiently small,
        2 = ||A'(Ax - b)|| sufficiently small,
        7 = iteration limit reached.
    itn : int
        Number of iterations performed.
    """
    m, n = A.shape
    device = b.device
    dtype = b.dtype

    if maxiter is None:
        maxiter = min(m, n)

    # Initialize
    x = torch.zeros(n, dtype=dtype, device=device)

    # Golub-Kahan bidiagonalization initialization
    # beta_1 * u_1 = b
    beta = torch.linalg.norm(b).item()
    if beta == 0.0:
        return x, 0, 0
    u = b / beta

    # alpha_1 * v_1 = A' * u_1
    v = torch.mv(A.t(), u)
    alpha = torch.linalg.norm(v).item()
    if alpha == 0.0:
        return x, 0, 0
    v = v / alpha

    # Initialize variables for LSMR
    alpha_bar = alpha
    zeta_bar = alpha * beta
    rho = 1.0
    rho_bar = 1.0
    c_bar = 1.0
    s_bar = 0.0
    zeta = 0.0

    h = v.clone()
    h_bar = torch.zeros(n, dtype=dtype, device=device)

    # Norm estimates
    norm_A2 = alpha * alpha
    norm_b = beta

    istop = 0
    itn = 0

    for itn in range(1, maxiter + 1):
        # Golub-Kahan step
        u = torch.mv(A, v) - alpha * u
        beta = torch.linalg.norm(u).item()
        if beta > 0.0:
            u = u / beta

        v = torch.mv(A.t(), u) - beta * v
        alpha = torch.linalg.norm(v).item()
        if alpha > 0.0:
            v = v / alpha

        # Construct rotation Q_hat (for damping)
        _chat, _shat, alpha_hat = _sym_ortho(alpha_bar, damp)

        # Plane rotation P_{k-1,k}
        rho_old = rho
        c, s, rho = _sym_ortho(alpha_hat, beta)
        theta_new = s * alpha
        alpha_bar = c * alpha

        # Plane rotation P_bar_{k-1,k}
        theta_bar = s_bar * rho
        rho_bar_old = rho_bar
        c_bar, s_bar, rho_bar = _sym_ortho(c_bar * rho, theta_new)
        zeta = c_bar * zeta_bar
        zeta_bar = -s_bar * zeta_bar

        # Update h, h_bar, x
        h_bar = h - (theta_bar * rho / (rho_old * rho_bar_old)) * h_bar
        x = x + (zeta / (rho * rho_bar)) * h_bar
        h = v - (theta_new / rho) * h

        # Estimate ||A||_F
        norm_A2 += beta * beta + alpha * alpha
        norm_A = sqrt(norm_A2)

        # Compute norms for stopping criteria (Fong & Saunders 2011, Section 5.3)
        norm_r = abs(zeta_bar)
        norm_Ar = abs(alpha_bar * zeta)
        norm_x = torch.linalg.norm(x).item()

        # Test for convergence
        if norm_b > 0:
            test1 = norm_r / norm_b
        else:
            test1 = 0.0

        if norm_A * norm_r > 0:
            test2 = norm_Ar / (norm_A * norm_r)
        else:
            test2 = 0.0

        rtol = btol + atol * norm_A * norm_x / norm_b if norm_b > 0 else atol
        if test1 <= rtol:
            istop = 1
            break
        if test2 <= atol:
            istop = 2
            break

    else:
        istop = 7  # iteration limit reached

    return x, istop, itn


def _sym_ortho(a: float, b: float) -> tuple[float, float, float]:
    """
    Compute a symmetric Givens rotation.

    Given a and b, returns (c, s, r) such that
    [c  s] [a] = [r]
    [-s c] [b]   [0]
    """
    if b == 0.0:
        if a == 0.0:
            c = 1.0
        else:
            c = float(np.sign(a))
        s = 0.0
        r = abs(a)
    elif a == 0.0:
        c = 0.0
        s = float(np.sign(b))
        r = abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = float(np.sign(b)) / sqrt(1.0 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = float(np.sign(a)) / sqrt(1.0 + tau * tau)
        s = c * tau
        r = a / c
    return c, s, r


class PyTorchFWLDemeaner:
    """
    Frisch-Waugh-Lovell theorem demeaner using sparse LSMR solver in PyTorch.

    Mirrors CupyFWLDemeaner but uses a hand-rolled LSMR implementation
    operating on PyTorch sparse CSR tensors.
    """

    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        solver_atol: float = 1e-8,
        solver_btol: float = 1e-8,
        solver_maxiter: Optional[int] = None,
        warn_on_cpu_fallback: bool = True,
        dtype: type = np.float64,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the pytorch backend.")

        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        elif use_gpu and not torch.cuda.is_available():
            if warn_on_cpu_fallback:
                warnings.warn(
                    "CUDA not available. Falling back to CPU. "
                    "Install PyTorch with CUDA support for GPU acceleration.",
                    UserWarning,
                )
            self.use_gpu = False
        else:
            self.use_gpu = use_gpu

        self.solver_atol = solver_atol
        self.solver_btol = solver_btol
        self.solver_maxiter = solver_maxiter
        self.warn_on_cpu_fallback = warn_on_cpu_fallback
        self.dtype = dtype
        self.torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

    def _solve_lsmr_loop(
        self,
        D_weighted: "torch.Tensor",
        x_weighted: "torch.Tensor",
        D_unweighted: "torch.Tensor",
        x_unweighted: "torch.Tensor",
    ) -> tuple["torch.Tensor", bool]:
        """Solve OLS equations via LSMR solver."""
        X_k = x_unweighted.shape[1]
        D_k = D_weighted.shape[1]
        theta = torch.zeros(D_k, X_k, dtype=x_unweighted.dtype, device=x_unweighted.device)
        success = True

        for k in range(X_k):
            result_x, istop, _ = lsmr_pytorch(
                D_weighted,
                x_weighted[:, k],
                damp=0.0,
                atol=self.solver_atol,
                btol=self.solver_btol,
                maxiter=self.solver_maxiter,
            )
            theta[:, k] = result_x
            success = success and (istop in [1, 2, 3])

        x_demeaned = x_unweighted - torch.sparse.mm(D_unweighted, theta)

        return x_demeaned, success

    def demean(
        self,
        x: NDArray[Any],
        flist: NDArray[Any],
        weights: NDArray[Any],
        tol: float = 1e-8,
        maxiter: int = 100_000,
        fe_sparse_matrix: Optional[Any] = None,
    ) -> tuple[NDArray[Any], bool]:
        """
        Demean variable x by projecting out fixed effects using FWL theorem.

        Parameters
        ----------
        x : np.ndarray
            Variable(s) to demean.
        flist : np.ndarray
            Integer-encoded fixed effects. Ignored if fe_sparse_matrix provided.
        weights : np.ndarray, shape (n_obs,)
            Weights (1.0 for equal weighting).
        tol : float, default=1e-8
            Convergence tolerance. Used for both atol and btol.
        maxiter : int, default=100_000
            Maximum iterations for LSMR.
        fe_sparse_matrix : scipy.sparse.csr_matrix, optional
            Pre-computed sparse FE dummy matrix.

        Returns
        -------
        x_demeaned : np.ndarray
            Demeaned variable (residuals after projecting out FEs).
        success : bool
            True if solver converged/succeeded.
        """
        if self.solver_maxiter is None:
            self.solver_maxiter = maxiter

        if fe_sparse_matrix is None:
            raise ValueError("fe_sparse_matrix cannot be None")

        # Convert to torch tensors on target device
        D = _convert_scipy_sparse_to_torch(fe_sparse_matrix, self.torch_dtype, self.device)
        x_device = torch.from_numpy(x.astype(self.dtype, copy=False)).to(
            dtype=self.torch_dtype, device=self.device
        )
        weights_device = torch.from_numpy(weights.astype(self.dtype, copy=False)).to(
            dtype=self.torch_dtype, device=self.device
        )

        # Apply sqrt-weight transform
        if weights is not None:
            sqrt_w = torch.sqrt(weights_device)
            if x_device.ndim == 2:
                x_weighted = x_device * sqrt_w[:, None]
            else:
                x_weighted = x_device * sqrt_w
            # Multiply sparse matrix rows by sqrt_w
            D_weighted = _sparse_row_scale(D, sqrt_w)
        else:
            x_weighted = x_device
            D_weighted = D

        x_demeaned, success = self._solve_lsmr_loop(
            D_weighted, x_weighted, D, x_device
        )

        # Convert back to numpy float64
        if self.torch_dtype == torch.float64:
            result = x_demeaned.cpu().numpy()
        else:
            result = x_demeaned.to(torch.float64).cpu().numpy()

        return result, success


def _sparse_row_scale(sparse_csr: "torch.Tensor", scale: "torch.Tensor") -> "torch.Tensor":
    """Scale rows of a sparse CSR matrix by a dense vector (vectorized)."""
    crow_indices = sparse_csr.crow_indices()
    col_indices = sparse_csr.col_indices()
    values = sparse_csr.values().clone()

    # Compute number of nonzeros per row, then repeat scale values accordingly
    nnz_per_row = crow_indices[1:] - crow_indices[:-1]
    row_scale = torch.repeat_interleave(scale, nnz_per_row)
    values *= row_scale

    return torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=sparse_csr.shape,
        dtype=sparse_csr.dtype,
        device=sparse_csr.device,
    )


def demean_pytorch(
    x: NDArray[np.float64],
    flist: Optional[NDArray[np.uint64]] = None,
    weights: Optional[NDArray[np.float64]] = None,
    tol: float = 1e-8,
    maxiter: int = 100_000,
) -> tuple[NDArray[np.float64], bool]:
    """
    PyTorch demeaner using float64 precision with CUDA auto-detection.

    Parameters
    ----------
    x : np.ndarray
        Variable(s) to demean.
    flist : np.ndarray
        Integer-encoded fixed effects.
    weights : np.ndarray, optional
        Observation weights. Defaults to equal weights.
    tol : float, default=1e-8
        Convergence tolerance.
    maxiter : int, default=100_000
        Maximum LSMR iterations.

    Returns
    -------
    x_demeaned : np.ndarray
        Demeaned variables.
    success : bool
        True if solver converged.
    """
    if weights is None:
        weights = np.ones(x.shape[0] if x.ndim > 1 else len(x))

    if flist is None:
        raise ValueError("flist cannot be None")

    n_fe = flist.shape[1] if flist.ndim > 1 else 1
    fe_df = pd.DataFrame(flist, columns=[f"f{i + 1}" for i in range(n_fe)], copy=False)
    fe_sparse_matrix = create_fe_sparse_matrix(fe_df)

    return PyTorchFWLDemeaner(dtype=np.float64).demean(
        x, flist, weights, tol, maxiter, fe_sparse_matrix=fe_sparse_matrix
    )
