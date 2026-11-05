"""
Newton-Raphson AC load flow, vectorised over many time steps at once.

The admittance matrix does not change over time (fixed topology and taps), so
all hours can be iterated together: V has shape (T, n) and the Jacobian
(T, 2m, 2m) is solved with one batched np.linalg.solve. Long periods are split
into chunks so memory stays bounded. Only numpy is needed, so this also runs
in the browser (Pyodide), where lightsim2grid is not available.
"""
import numpy as np


class PowerFlowError(RuntimeError):
    pass


def _chunk_size(n_pq, n):
    per_hour = (2 * n_pq) ** 2 * 8 * 3 + n * n * 16 * 4     # Jacobian + temporaries, bytes
    return int(max(1, min(8760, 3e8 // max(per_hour, 1))))


def solve(net, P, Q, tol=1e-8, max_iter=30):
    """P, Q: (T, n) net injections in MW / Mvar per (internal) bus.
    Returns dict with vm_pu (T, n_bus), line_loading (T, n_line) [%],
    trafo_loading (T, n_trafo) [%], slack_p_mw (T,) and the complex voltages."""
    T = P.shape[0]
    n = net.n
    S = (P + 1j * Q) / net.sn_mva
    V = np.empty((T, n), complex)
    step = _chunk_size(len(net.pq), n)
    bad = []
    for a in range(0, T, step):
        b = min(T, a + step)
        V[a:b], badc = _nr(net, S[a:b], tol, max_iter)
        bad.extend((a + i) for i in badc)
    if bad:
        raise PowerFlowError(
            f"The load flow did not converge in {len(bad)} of {T} hours (first: hour {bad[0]}). "
            "The network is probably overloaded or a line/transformer parameter is far off.")
    return _results(net, V)


def _nr(net, S, tol, max_iter):
    Y, pq, ref = net.Y, net.pq, net.ref
    T, n = S.shape
    m = len(pq)
    V = np.tile(net.V_init, (T, 1))
    if m == 0:
        return V, []
    idx = np.arange(n)
    YT = Y.T
    for _ in range(max_iter):
        I = V @ YT
        mis = V * np.conj(I) - S
        F = np.concatenate([mis[:, pq].real, mis[:, pq].imag], axis=1)
        err = np.max(np.abs(F), axis=1)
        if not np.isfinite(err).all():
            break
        if err.max() < tol:
            return V, []
        act = err >= tol                                 # only iterate unconverged hours
        Va, Ia = V[act], I[act]
        Vn = Va / np.abs(Va)
        dVm = Va[:, :, None] * np.conj(Y[None] * Vn[:, None, :])
        dVm[:, idx, idx] += np.conj(Ia) * Vn
        M = -(Y[None] * Va[:, None, :])
        M[:, idx, idx] += Ia
        dVa = 1j * Va[:, :, None] * np.conj(M)
        dVa = dVa[:, pq][:, :, pq]; dVm = dVm[:, pq][:, :, pq]
        J = np.block([[dVa.real, dVm.real], [dVa.imag, dVm.imag]])
        try:
            dx = np.linalg.solve(J, -F[act][:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            break
        ang = np.angle(Va); mag = np.abs(Va)
        ang[:, pq] += dx[:, :m]; mag[:, pq] += dx[:, m:]
        V[act] = mag * np.exp(1j * ang)
    I = V @ YT
    mis = V * np.conj(I) - S
    err = np.max(np.abs(np.concatenate([mis[:, pq].real, mis[:, pq].imag], axis=1)), axis=1)
    return V, np.nonzero(~(err < tol * 10))[0].tolist()


def _results(net, V):
    If = V @ net.Yf.T
    It = V @ net.Yt.T
    aF = np.abs(If) * net.i_scale_f
    aT = np.abs(It) * net.i_scale_t
    lb, tb = net.line_br, net.trafo_br
    line_loading = np.maximum(aF[:, lb], aT[:, lb]) / net.line_imax * 100.0
    trafo_loading = np.maximum(aF[:, tb] / net.trafo_in_hv, aT[:, tb] / net.trafo_in_lv) * 100.0
    I = V @ net.Y.T
    S_bus = V * np.conj(I) * net.sn_mva
    return {
        "V": V,
        "vm_pu": np.abs(V[:, :net.n_bus]),
        "line_loading": line_loading,
        "trafo_loading": trafo_loading,
        "loading": np.concatenate([line_loading, trafo_loading], axis=1),   # same order as net.elements
        "trafo_p_hv_mw": (V[:, net.f[tb]] * np.conj(If[:, tb])).real * net.sn_mva,
        "slack_p_mw": S_bus[:, net.ref].real.sum(axis=1),
    }


def injections(net, load_scale, sgen_scale, extra_bus=None, extra_p=None):
    """Net bus injections (T, n) from per-element scale factors.
    load_scale: (T, n_load), sgen_scale: (T, n_sgen).
    extra_bus/extra_p: optional battery (internal bus index, (T,) MW, + = discharge)."""
    T = load_scale.shape[0]
    P = np.zeros((T, net.n)); Q = np.zeros((T, net.n))
    if len(net.loads):
        np.add.at(P.T, net.load_bus, -(load_scale * net.load_p).T)
        np.add.at(Q.T, net.load_bus, -(load_scale * net.load_q).T)
    if len(net.sgens):
        np.add.at(P.T, net.sgen_bus, (sgen_scale * net.sgen_p).T)
        np.add.at(Q.T, net.sgen_bus, (sgen_scale * net.sgen_q).T)
    if extra_bus is not None:
        P[:, extra_bus] += extra_p
    return P, Q
