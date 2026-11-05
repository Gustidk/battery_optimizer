"""
Battery schedule optimisation -- the same formulations as the original
control/battery_optimizer.py (sensitivity model) and
control/piecewise_optimizer.py (piecewise linear model), but with sparse
matrices (a full year fits in a few MB) and solvers that also run in the
browser: HiGHS (highspy) for the LPs, Clarabel for the QPs.

Variables: x = [discharge (T) | charge (T) | soc (T) | peak (L)]
"""
import numpy as np
import scipy.sparse as sp


class OptimisationError(RuntimeError):
    pass


# ----------------------------------------------------------------------- solvers
LP_OPTIONS = {}


def solve_lp(c, A_ub, b_ub, A_eq, b_eq, lb, ub, options=None):
    import highspy
    inf = highspy.kHighsInf
    A = sp.vstack([A_eq, A_ub], format="csc")
    lo = np.r_[b_eq, np.full(A_ub.shape[0], -inf)]
    hi = np.r_[b_eq, b_ub]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    for k, v in {**LP_OPTIONS, **(options or {})}.items():
        h.setOptionValue(k, v)
    lp = highspy.HighsLp()
    lp.num_col_ = len(c); lp.num_row_ = A.shape[0]
    lp.col_cost_ = np.asarray(c, float)
    lp.col_lower_ = np.where(np.isfinite(lb), lb, -inf)
    lp.col_upper_ = np.where(np.isfinite(ub), ub, inf)
    lp.row_lower_ = lo; lp.row_upper_ = hi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    h.passModel(lp)
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    if status != "Optimal":
        raise OptimisationError(f"The LP solver stopped with status '{status}'.")
    return np.array(h.getSolution().col_value)


def solve_qp(P_diag_idx, P_diag_val, q, A_ub, b_ub, A_eq, b_eq, lb, ub):
    """min 0.5 x'Px + q'x with diagonal P (only on given indices)."""
    import clarabel
    n = len(q)
    P = sp.csc_matrix((P_diag_val, (P_diag_idx, P_diag_idx)), shape=(n, n))
    I = sp.identity(n, format="csc")
    fu = np.isfinite(ub); fl = np.isfinite(lb)
    A = sp.vstack([A_eq, A_ub, I[fu], -I[fl]], format="csc")
    b = np.r_[b_eq, b_ub, ub[fu], -lb[fl]]
    cones = [clarabel.ZeroConeT(A_eq.shape[0]), clarabel.NonnegativeConeT(A.shape[0] - A_eq.shape[0])]
    s = clarabel.DefaultSettings()
    s.verbose = False
    s.max_iter = 200
    sol = clarabel.DefaultSolver(P, np.asarray(q, float), A, b, cones, s).solve()
    st = str(sol.status)
    if "Solved" not in st:          # Solved or AlmostSolved
        raise OptimisationError(f"The QP solver stopped with status '{st}'.")
    return np.array(sol.x)


# ------------------------------------------------------------------ building blocks
def soc_equations(T, n, battery_mwh, eta_c, eta_d, soc_init_frac, dt_h):
    """soc[t] - soc[t-1] - eta_c*dt*c[t] + dt/eta_d*d[t] = 0, cyclic: soc[T-1] = soc0."""
    soc0 = soc_init_frac * battery_mwh
    t = np.arange(T)
    rows = np.r_[t, t, t, t[1:], T]
    cols = np.r_[2 * T + t, T + t, t, 2 * T + t[1:] - 1, 3 * T - 1]
    vals = np.r_[np.ones(T), np.full(T, -eta_c * dt_h), np.full(T, dt_h / eta_d), -np.ones(T - 1), 1.0]
    A = sp.csc_matrix((vals, (rows, cols)), shape=(T + 1, n))
    b = np.zeros(T + 1); b[0] = soc0; b[T] = soc0
    return A, b


def element_weights(net, weight_line, weight_trafo, element_weights=None):
    w = np.array([weight_trafo if e[0] == "trafo" else weight_line for e in net.elements], float)
    if element_weights:
        for k, e in enumerate(net.elements):
            key = f"{e[0]}:{e[1]}"
            if key in element_weights:
                w[k] = float(element_weights[key])
    return w


def _unpack(x, T, L):
    return x[:T], x[T:2 * T], x[2 * T:3 * T], x[3 * T:3 * T + L]


# --------------------------------------------------- sensitivity (tangent) model
def sensitivity_rows(baseline, s_d, s_c, mw):
    """peak[l] >= baseline[t,l] + s_d[t,l]*d[t] + s_c[t,l]*c[t]   ->   A_ub x <= b_ub."""
    T, L = baseline.shape
    s_d = np.broadcast_to(s_d, (T, L)); s_c = np.broadcast_to(s_c, (T, L))
    n = 3 * T + L
    # Exact row elimination: row (t,l) can never bind if the highest value it can take
    # (battery at full power in its worst direction) is below the lowest value some other
    # hour of the same element is guaranteed to have. Peak[l] is >= that value anyway.
    hi = baseline + (np.maximum(s_d, 0) + np.maximum(s_c, 0)) * mw
    lo = baseline + (np.minimum(s_d, 0) + np.minimum(s_c, 0)) * mw
    keep = hi > lo.max(axis=0)[None, :] - 1e-9
    tt, ll = np.nonzero(keep)
    nr = len(tt)
    r = np.arange(nr)
    A = sp.csc_matrix((np.r_[-np.ones(nr), s_d[tt, ll], s_c[tt, ll]],
                       (np.r_[r, r, r], np.r_[3 * T + ll, tt, T + tt])), shape=(nr, n))
    return A, -baseline[tt, ll]



def optimise_sensitivity(objective, baseline, s_d, s_c, bat, weights=None):
    """objective 'minmax' (LP, objective 1) or 'quadratic_peak' (QP, objective 3)."""
    T, L = baseline.shape
    n = 3 * T + L
    A_eq, b_eq = soc_equations(T, n, bat["mwh"], bat["eta_c"], bat["eta_d"], bat["soc_init"], bat["dt_h"])
    A_ub, b_ub = sensitivity_rows(baseline, s_d, s_c, bat["mw"])
    ub = np.r_[np.full(2 * T, bat["mw"]), np.full(T, bat["mwh"]), np.full(L, np.inf)]
    if objective == "minmax":
        lb = np.r_[np.zeros(3 * T), np.full(L, -np.inf)]
        c = np.zeros(n); c[3 * T:] = 1.0 / L
        x = solve_lp(c, A_ub, b_ub, A_eq, b_eq, lb, ub)
    else:
        lb = np.zeros(n)
        x = solve_qp(np.arange(3 * T, n), 2.0 * weights, np.zeros(n), A_ub, b_ub, A_eq, b_eq, lb, ub)
    return _unpack(x, T, L)


# ---------------------------------------------------------- piecewise linear model
def piecewise_rows(table, levels, flat_tol=1e-3):
    """table: (T, K, L) AC loading at each power level. One row per chord:
    peak[l] >= intercept + slope*(d[t] - c[t]).  Flat hours only raise peak's floor."""
    T, K, L = table.shape
    n = 3 * T + L
    dp = np.diff(levels)
    slope = np.diff(table, axis=1) / dp[None, :, None]               # (T, K-1, L)
    icpt = table[:, :-1, :] - slope * levels[None, :-1, None]
    floor = np.zeros(L)
    rows, cols, vals, rhs = [], [], [], []
    nrow = 0
    for l in range(L):
        s_l = slope[:, :, l]
        flat = np.max(np.abs(s_l), axis=1) < flat_tol
        if flat.any():
            floor[l] = max(floor[l], float(table[flat, :, l].max()))
        act_all = np.nonzero(~flat)[0]
        if len(act_all) == 0:
            continue
        # exact row elimination (see sensitivity_rows): f_t(p) >= min over levels of the
        # table, so peak[l] >= max_t min_k table; a chord whose largest value on
        # [-mw, mw] is below that can never bind.
        guaranteed = max(float(table[:, :, l].min(axis=1).max()), floor[l])
        for k in range(K - 1):
            top = np.maximum(icpt[act_all, k, l] + slope[act_all, k, l] * levels[0],
                             icpt[act_all, k, l] + slope[act_all, k, l] * levels[-1])
            act = act_all[top > guaranteed - 1e-9]
            na = len(act)
            if na == 0:
                continue
            s = s_l[act, k]
            rr = nrow + np.arange(na)
            rows += [rr, rr, rr]; cols += [act, T + act, np.full(na, 3 * T + l)]
            vals += [s, -s, -np.ones(na)]
            rhs.append(-icpt[act, k, l])
            nrow += na
    if nrow:
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nrow, n))
        b = np.concatenate(rhs)
    else:
        A = sp.csc_matrix((0, n)); b = np.zeros(0)
    return A, b, floor


def optimise_piecewise(objective, table, levels, bat, weights=None,
                       eps_lp=1e-4, eps_qp=1e-2, mode_tol=1e-3):
    """Two passes (as in the original): pass 1 free, pass 2 with each hour's
    mode (charge / discharge / idle) locked, so the model SOC equals the physical SOC."""
    T, K, L = table.shape
    n = 3 * T + L
    A_eq, b_eq = soc_equations(T, n, bat["mwh"], bat["eta_c"], bat["eta_d"], bat["soc_init"], bat["dt_h"])
    A_ub, b_ub, floor = piecewise_rows(table, levels)

    def one_pass(max_d, max_c):
        ub = np.r_[max_d, max_c, np.full(T, bat["mwh"]), np.full(L, np.inf)]
        if objective == "minmax":
            lb = np.r_[np.zeros(3 * T), floor]
            c = np.zeros(n); c[3 * T:] = 1.0 / L; c[:2 * T] = eps_lp
            # interior point (Clarabel) -- much faster than simplex for this large LP,
            # also in the browser; pass 2 removes any ambiguity in charge/discharge
            return solve_qp(np.zeros(0, int), np.zeros(0), c, A_ub, b_ub, A_eq, b_eq, lb, ub)
        lb = np.r_[np.zeros(3 * T), np.maximum(floor, 0)]
        q = np.zeros(n); q[:2 * T] = eps_qp
        return solve_qp(np.arange(3 * T, n), 2.0 * weights, q, A_ub, b_ub, A_eq, b_eq, lb, ub)

    full = np.full(T, bat["mw"])
    x1 = one_pass(full, full)
    net1 = x1[:T] - x1[T:2 * T]
    x2 = one_pass(np.where(net1 > mode_tol, bat["mw"], 0.0), np.where(net1 < -mode_tol, bat["mw"], 0.0))
    return _unpack(x2, T, L)
