import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg

def estimate_chevpol_order(XYZ, rho, wl, eps):
    r"""
    Compute order of polynomial filter to approximate asymptotic
    point-spread function on \cS^{2}.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian instrument coordinates.
    rho : float
        Scale parameter \rho corresponding to the average distance of a point
        on the graph to its nearest neighbor.
        Output of :py:func:`~deepwave.tools.math.graph.laplacian_exp`.
    wl : float
        Wavelength of observations [m].
    eps : float
        Ratio in (0, 1).
        Ensures all PSF magnitudes lower than `max(PSF)*eps` past the main
        lobe are clipped at 0.

    Returns
    -------
    K : int
        Order of polynomial filter.
    """
    XYZ = XYZ / wl
    XYZ_centroid = np.mean(XYZ, axis=1, keepdims=True)
    XYZ_radius = np.mean(linalg.norm(XYZ - XYZ_centroid, axis=0))

    theta = np.linspace(0, np.pi, 1000)
    f = 20 * np.log10(np.abs(np.sinc(theta / np.pi)))
    eps_dB = 10 * np.log10(eps)
    theta_max = np.max(theta[f >= eps_dB])

    beam_width = theta_max / (2 * np.pi * XYZ_radius)
    K = np.sqrt(2 - 2 * np.cos(beam_width)) / rho
    K = int(np.ceil(K))
    return K

def steering_operator(XYZ, R, wl):
    '''
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian array geometry.
    R : :py:class:`~numpy.ndarray`
        (3, N_px) Cartesian grid points in :math:`\mathbb{S}^{2}`.
    wl : float
        Wavelength [m].
    return: steering matrix
    '''
    scale = 2 * np.pi / wl
    A = np.exp((-1j * scale * XYZ.T) @ R)
    return A

def eighMax(A):
    r"""
    Evaluate :math:`\mu_{\max}(\bbB)` with
    :math:
    B = (\overline{\bbA} \circ \bbA)^{H} (\overline{\bbA} \circ \bbA)
    Uses a matrix-free formulation of the Lanczos algorithm.
    Parameters
    ----------
    A : :py:class:`~numpy.ndarray`
        (M, N) array.

    Returns
    -------
    D_max : float
        Leading eigenvalue of `B`.
    """
    if A.ndim != 2:
        raise ValueError('Parameter[A] has wrong dimensions.')

    def matvec(v):
        r"""
        Parameters
        ----------
        v : :py:class:`~numpy.ndarray`
            (N,) or (N, 1) array

        Returns
        -------
        w : :py:class:`~numpy.ndarray`
            (N,) array containing :math:`\bbB \bbv`
        """
        v = v.reshape(-1)

        C = (A * v) @ A.conj().T
        D = C @ A
        w = np.sum(A.conj() * D, axis=0).real
        return w

    M, N = A.shape
    B = splinalg.LinearOperator(shape=(N, N),
                                matvec=matvec,
                                dtype=np.float64)
    D_max = splinalg.eigsh(B, k=1, which='LM', return_eigenvectors=False)
    return D_max[0]