import numpy as np

_DIM = 2
_N_SAMPLES = 1000
_BASE_SEED = 20260327
_EPS = 1e-300
_PDF_FLOOR = 1e-300
_EVAL_BETA_OFFSETS = (0.0,)
_N_BLOCKS = 4
_PILOT_CAL_N = 4096
_PILOT_LEVEL_PROBS = (0.20, 0.05, 0.01, 0.005)
_PILOT_CACHE = {}
_LEVEL_LOSS_WEIGHTS = (0.25, 0.5, 1.0, 1.5)
_W_FINAL_VARIANCE = 1.0
_W_LEVEL_VARIANCE = 0.10
_LEVEL_NAV_ESS_FRAC = 0.85
_W_FAILURE_UNDERCOVERAGE = 1.0
_W_FAILURE_SHAPE = 0.35
_FAILURE_LOG_RATIO_CAP = 4.0
_FAILURE_TAIL_FRAC = 0.05
_SCOUT_N = 2000
_SCOUT_RHO = 0.10
_SCOUT_LEVELS = 12
_SCOUT_MCMC_STEPS = 8
_SCOUT_MIN_FAILURES = 20
_SCOUT_MAX_FAILURES = 1500
_SCOUT_STEP_SCALE = 2.38 / np.sqrt(_DIM)
_SCOUT_CACHE = {}


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def _target_logpdf_standard(x: np.ndarray) -> np.ndarray:
    x = _as_2d(x)
    return -0.5 * _DIM * np.log(2.0 * np.pi) - 0.5 * np.sum(x * x, axis=1)


def target_pdf(x: np.ndarray) -> np.ndarray:
    x = _as_2d(x)
    return np.maximum(np.exp(_target_logpdf_standard(x)), _PDF_FLOOR)


def target_sampler(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=1.0, size=(n_samples, _DIM))


def limit_state(x: np.ndarray, beta: float) -> np.ndarray:
    x = _as_2d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]
    return 0.5 * (x1 - 2.0) ** 2 - 1.5 * (x2 - 5.0) ** 3 - beta


def adaptive_failure_scout(beta: float, seed: int) -> np.ndarray:
    beta = float(beta)
    cache_key = (round(beta, 12), int(seed))
    if cache_key in _SCOUT_CACHE:
        return _SCOUT_CACHE[cache_key].copy()

    rng = np.random.default_rng(seed)

    try:
        x = target_sampler(_SCOUT_N, rng)
        if x.shape != (_SCOUT_N, _DIM) or not np.all(np.isfinite(x)):
            _SCOUT_CACHE[cache_key] = np.empty((0, _DIM), dtype=float)
            return _SCOUT_CACHE[cache_key].copy()

        g = limit_state(x, beta)
    except Exception:
        _SCOUT_CACHE[cache_key] = np.empty((0, _DIM), dtype=float)
        return _SCOUT_CACHE[cache_key].copy()

    archive = []

    for _ in range(_SCOUT_LEVELS):
        if len(x) == 0 or not np.all(np.isfinite(g)):
            break

        fail_mask = g <= 0.0
        if np.any(fail_mask):
            archive.append(x[fail_mask])
            if sum(len(a) for a in archive) >= _SCOUT_MIN_FAILURES:
                break

        gamma = float(np.quantile(g, _SCOUT_RHO))
        if gamma < 0.0:
            gamma = 0.0

        elite_mask = g <= gamma
        elites = x[elite_mask]
        if len(elites) == 0:
            break

        idx = rng.integers(0, len(elites), size=_SCOUT_N)
        x_new = elites[idx].copy()
        g_new = limit_state(x_new, beta)
        logp_new = _target_logpdf_standard(x_new)

        for _ in range(_SCOUT_MCMC_STEPS):
            proposal = x_new + _SCOUT_STEP_SCALE * rng.normal(size=x_new.shape)

            try:
                g_prop = limit_state(proposal, beta)
            except Exception:
                continue

            ok = np.isfinite(g_prop) & (g_prop <= gamma)
            if not np.any(ok):
                continue

            logp_prop = _target_logpdf_standard(proposal)
            log_alpha = logp_prop - logp_new
            accept = ok & (np.log(rng.random(_SCOUT_N) + _EPS) < log_alpha)

            if np.any(accept):
                x_new[accept] = proposal[accept]
                g_new[accept] = g_prop[accept]
                logp_new[accept] = logp_prop[accept]

        x = x_new
        g = g_new

    if len(archive) == 0:
        zf = np.empty((0, _DIM), dtype=float)
    else:
        zf = np.vstack(archive)
        zf = zf[np.all(np.isfinite(zf), axis=1)]
        if len(zf) > _SCOUT_MAX_FAILURES:
            keep = rng.choice(len(zf), size=_SCOUT_MAX_FAILURES, replace=False)
            zf = zf[keep]

    _SCOUT_CACHE[cache_key] = zf.copy()
    return zf


def _target_log_prob(x: np.ndarray) -> np.ndarray:
    x = _as_2d(x)
    out = _target_logpdf_standard(x)
    out = np.asarray(out, dtype=float).reshape(-1)
    if out.shape != (len(x),) or np.any(~np.isfinite(out)):
        raise ValueError("invalid target log-density")
    return out


def _proposal_log_prob(pdf, x: np.ndarray) -> np.ndarray:
    x = _as_2d(x)
    q = np.asarray(pdf(x), dtype=float).reshape(-1)
    if q.shape != (len(x),) or np.any(~np.isfinite(q)) or np.any(q <= 0.0):
        raise ValueError("invalid proposal density")
    out = np.log(np.maximum(q, _PDF_FLOOR))
    out = np.asarray(out, dtype=float).reshape(-1)
    if out.shape != (len(x),) or np.any(~np.isfinite(out)):
        raise ValueError("invalid proposal log-density")
    return out


def _logsumexp_1d(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    if len(a) == 0:
        return -np.inf
    m = float(np.max(a))
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(a - m))))


def _event_stats_from_logweights(logw: np.ndarray, event_mask: np.ndarray, n_samples: int):
    logw = np.asarray(logw, dtype=float).reshape(-1)
    mask = np.asarray(event_mask, dtype=bool).reshape(-1)
    n = int(n_samples)
    if logw.shape != mask.shape or len(logw) != n:
        raise ValueError("event/logweight shape mismatch")

    idx = np.flatnonzero(mask)
    n_event = int(len(idx))
    if n_event == 0:
        return {
            "prob_hat": 0.0,
            "log_prob_hat": -np.inf,
            "internal_cv2": float("inf"),
            "ess": 0.0,
            "max_contrib_share": 1.0,
            "n_event": 0,
        }

    logy = logw[idx]
    if np.any(np.isnan(logy)):
        raise ValueError("NaN importance log weights")

    log_sum_y = _logsumexp_1d(logy)
    log_sum_y2 = _logsumexp_1d(2.0 * logy)
    log_prob_hat = float(log_sum_y - np.log(float(n)))
    prob_hat = float(np.exp(log_prob_hat)) if -745.0 < log_prob_hat < 700.0 else (
        0.0 if log_prob_hat <= -745.0 else float("inf")
    )

    log_ess = 2.0 * log_sum_y - log_sum_y2
    ess = float(np.exp(np.clip(log_ess, -745.0, 700.0)))
    ess = min(max(ess, 0.0), float(n_event))
    max_share = float(np.exp(np.max(logy) - log_sum_y))

    if n > 1:
        log_ratio = log_sum_y2 - 2.0 * log_sum_y
        ratio = float(np.exp(min(log_ratio, 700.0)))
        internal_cv2 = max(0.0, (n * ratio - 1.0) / (n - 1.0))
    else:
        internal_cv2 = float("inf")

    return {
        "prob_hat": prob_hat,
        "log_prob_hat": log_prob_hat,
        "internal_cv2": internal_cv2,
        "ess": ess,
        "max_contrib_share": max_share,
        "n_event": n_event,
    }


def _normalized_relative_variance_loss(cv2: float, n_samples: int) -> float:
    n = int(n_samples)
    if n <= 1:
        return 1.0
    cv2 = float(cv2)
    if not np.isfinite(cv2):
        return 1.0
    cv2 = max(cv2, 0.0)
    value = np.log1p((n - 1.0) * cv2) / np.log(float(n))
    return float(np.clip(value, 0.0, 1.0))


def _split_block_sizes(total: int, n_blocks: int):
    total = int(total)
    n_blocks = max(1, min(int(n_blocks), total))
    q, r = divmod(total, n_blocks)
    return [q + (1 if j < r else 0) for j in range(n_blocks)]


def _draw_candidate_blocks(sampler, pdf, beta: float, n_samples: int):
    base = _BASE_SEED + int(round(1_000_000.0 * float(beta))) + 1_000_003 * int(_DIM)
    blocks = []
    for block_id, nb in enumerate(_split_block_sizes(n_samples, _N_BLOCKS)):
        rng = np.random.default_rng(base + 104729 * (block_id + 1))
        x = _as_2d(sampler(nb, rng))
        if x.shape != (nb, int(_DIM)) or np.any(~np.isfinite(x)):
            raise ValueError("proposal returned invalid samples")
        logq = _proposal_log_prob(pdf, x)
        logp = _target_log_prob(x)
        g = np.asarray(limit_state(x, beta), dtype=float).reshape(-1)
        if g.shape != (nb,) or np.any(~np.isfinite(g)):
            raise ValueError("invalid limit-state output")
        logw = logp - logq
        if np.any(np.isnan(logw)):
            raise ValueError("NaN importance log weights")
        blocks.append({"x": x, "logw": logw, "g": g, "n": nb})
    return blocks


def get_pilot_data(beta: float):
    key = (int(_DIM), float(beta), int(_PILOT_CAL_N), tuple(_PILOT_LEVEL_PROBS))
    if key in _PILOT_CACHE:
        return _PILOT_CACHE[key]

    seed = _BASE_SEED + 15485863 + 1_000_003 * int(_DIM) + int(round(1_000_000.0 * float(beta)))
    rng = np.random.default_rng(seed)
    x = _as_2d(target_sampler(int(_PILOT_CAL_N), rng))
    if x.shape != (int(_PILOT_CAL_N), int(_DIM)) or np.any(~np.isfinite(x)):
        raise ValueError("target_sampler returned invalid pilot samples")
    g = np.asarray(limit_state(x, beta), dtype=float).reshape(-1)
    if g.shape != (int(_PILOT_CAL_N),) or np.any(~np.isfinite(g)):
        raise ValueError("invalid pilot limit-state values")

    try:
        thresholds = np.quantile(g, np.asarray(_PILOT_LEVEL_PROBS, dtype=float), method="linear")
    except TypeError:
        thresholds = np.quantile(g, np.asarray(_PILOT_LEVEL_PROBS, dtype=float), interpolation="linear")

    deepest = float(thresholds[int(np.argmin(np.asarray(_PILOT_LEVEL_PROBS, dtype=float)))])
    elite_idx = np.flatnonzero(g <= deepest)
    if len(elite_idx) < 16:
        elite_idx = np.argsort(g, kind="mergesort")[:16]

    out = {
        "thresholds": np.asarray(thresholds, dtype=float),
        "deep_elite_x": x[elite_idx].copy(),
    }
    _PILOT_CACHE[key] = out
    return out


def _level_variance_loss(overall_logw: np.ndarray, overall_g: np.ndarray, thresholds) -> float:
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
    weights = np.asarray(_LEVEL_LOSS_WEIGHTS, dtype=float).reshape(-1)
    if len(weights) != len(thresholds):
        weights = np.ones(len(thresholds), dtype=float)
    weights = np.maximum(weights, 0.0)
    weights = weights / max(float(np.sum(weights)), _EPS)

    n = int(len(overall_g))
    losses = []
    for thr in thresholds:
        st = _event_stats_from_logweights(overall_logw, overall_g <= float(thr), n)
        losses.append(_normalized_relative_variance_loss(st["internal_cv2"], n))
    return float(np.clip(np.dot(weights, np.asarray(losses, dtype=float)), 0.0, 1.0))


def _upper_tail_mean(values: np.ndarray, frac: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0:
        return 1.0
    k = max(1, int(np.ceil(float(frac) * len(values))))
    if k >= len(values):
        return float(np.mean(values))
    idx = np.argpartition(values, len(values) - k)[-k:]
    return float(np.mean(values[idx]))


def _failure_domain_losses(pdf_aux, beta: float, pilot):
    seed = _BASE_SEED + 49979687 + 1_000_003 * int(_DIM) + int(round(1_000_000.0 * float(beta)))
    try:
        xf = _as_2d(adaptive_failure_scout(beta=beta, seed=seed))
    except Exception:
        xf = np.empty((0, int(_DIM)), dtype=float)

    if len(xf) >= int(_SCOUT_MIN_FAILURES):
        x = xf
        source = "failure_scout"
    else:
        x = _as_2d(pilot["deep_elite_x"])
        source = "deep_pilot_fallback"

    if len(x) == 0:
        return 1.0, 1.0, source, 0

    try:
        logp = _target_log_prob(x)
        logq = _proposal_log_prob(pdf_aux, x)
    except Exception:
        return 1.0, 1.0, source, int(len(x))

    log_ratio = np.nan_to_num(logq - logp, nan=-_FAILURE_LOG_RATIO_CAP, neginf=-_FAILURE_LOG_RATIO_CAP, posinf=_FAILURE_LOG_RATIO_CAP)
    deficit = np.maximum(-log_ratio, 0.0)
    deficit = np.clip(deficit / _FAILURE_LOG_RATIO_CAP, 0.0, 1.0)
    undercoverage = 0.25 * float(np.mean(deficit)) + 0.75 * _upper_tail_mean(deficit, _FAILURE_TAIL_FRAC)

    if source == "failure_scout":
        med = float(np.median(log_ratio))
        deviation = np.abs(log_ratio - med)
        deviation = np.clip(deviation / _FAILURE_LOG_RATIO_CAP, 0.0, 1.0)
        shape = 0.25 * float(np.mean(deviation)) + 0.75 * _upper_tail_mean(deviation, _FAILURE_TAIL_FRAC)
    else:
        shape = 0.0

    return float(np.clip(undercoverage, 0.0, 1.0)), float(np.clip(shape, 0.0, 1.0)), source, int(len(x))


def evaluate_candidate(sampler, pdf, pdf_aux, beta: float, n_samples: int):
    pilot = get_pilot_data(beta)
    blocks = _draw_candidate_blocks(sampler, pdf, beta, int(n_samples))
    all_logw = np.concatenate([b["logw"] for b in blocks])
    all_g = np.concatenate([b["g"] for b in blocks])

    final_stats = _event_stats_from_logweights(all_logw, all_g <= 0.0, int(n_samples))
    final_variance_loss = _normalized_relative_variance_loss(final_stats["internal_cv2"], int(n_samples))

    ess_cutoff = max(_LEVEL_NAV_ESS_FRAC * float(n_samples), _EPS)
    final_ess = float(final_stats["ess"])
    level_navigation_weight = float(_W_LEVEL_VARIANCE * np.clip(1.0 - final_ess / ess_cutoff, 0.0, 1.0))
    if level_navigation_weight > 0.0:
        level_variance_loss = _level_variance_loss(all_logw, all_g, pilot["thresholds"])
    else:
        level_variance_loss = 0.0

    undercoverage_loss, failure_shape_loss, coverage_source, coverage_points = _failure_domain_losses(pdf_aux, beta, pilot)

    loss = (
        _W_FINAL_VARIANCE * final_variance_loss
        + level_navigation_weight * level_variance_loss
        + _W_FAILURE_UNDERCOVERAGE * undercoverage_loss
        + _W_FAILURE_SHAPE * failure_shape_loss
    )

    return {
        "loss": float(loss),
        "pf_hat": float(final_stats["prob_hat"]),
        "internal_cv2": float(final_stats["internal_cv2"]),
        "ess_failure": float(final_stats["ess"]),
        "n_failure": int(final_stats["n_event"]),
        "max_contrib_share": float(final_stats["max_contrib_share"]),
        "final_variance_loss": float(final_variance_loss),
        "level_variance_loss": float(level_variance_loss),
        "level_navigation_weight": float(level_navigation_weight),
        "undercoverage_loss": float(undercoverage_loss),
        "failure_shape_loss": float(failure_shape_loss),
        "coverage_source": str(coverage_source),
        "coverage_points": int(coverage_points),
    }


def evaluate_for_one_beta(priority_fn, beta: float) -> float:
    sampler_aux, pdf_aux = priority_fn(beta)
    out = evaluate_candidate(
        sampler=sampler_aux,
        pdf=pdf_aux,
        pdf_aux=pdf_aux,
        beta=float(beta),
        n_samples=int(_N_SAMPLES),
    )
    return float(out["loss"])


def evaluate(priority_fn, beta: float) -> float:
    beta = float(beta)
    if not np.isfinite(beta):
        return 0.0

    losses = []
    for delta in _EVAL_BETA_OFFSETS:
        b = beta + float(delta)
        if not np.isfinite(b):
            losses.append(float("inf"))
            continue
        if b <= 0.0:
            losses.append(float("inf"))
            continue
        losses.append(evaluate_for_one_beta(priority_fn, b))

    total_loss = float(np.median(np.nan_to_num(np.asarray(losses, dtype=float), nan=1e12, posinf=1e12, neginf=1e12)))
    return float(np.clip(1.0 / (1.0 + total_loss), 0.0, 1.0))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / np.sum(ez)


def _mixture_pdf(x: np.ndarray, weights: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    x = _as_2d(x)
    k = means.shape[0]
    const = 1.0 / np.sqrt(2.0 * np.pi)
    density = np.zeros(len(x), dtype=float)
    for i in range(k):
        z = (x - means[i]) / scales[i]
        comp = np.prod(const * np.exp(-0.5 * z * z) / scales[i], axis=1)
        density += weights[i] * comp
    return np.maximum(density, _PDF_FLOOR)


def build_priority(params):
    logits = np.asarray(params["logits"], dtype=float).copy()
    means = np.asarray(params["means"], dtype=float).copy()
    log_scales = np.asarray(params["log_scales"], dtype=float).copy()

    def priority(beta: float):
        _ = float(beta)
        weights = _softmax(logits)
        scales = np.exp(log_scales)

        def sampler(n_samples: int, rng: np.random.Generator) -> np.ndarray:
            n_samples = int(n_samples)
            if n_samples <= 0:
                return np.empty((0, _DIM), dtype=float)
            ids = rng.choice(len(weights), size=n_samples, p=weights)
            noise = rng.normal(size=(n_samples, _DIM))
            samples = means[ids] + noise * scales[ids]
            return samples

        def pdf(x: np.ndarray) -> np.ndarray:
            return _mixture_pdf(x, weights=weights, means=means, scales=scales)

        return sampler, pdf

    return priority


def mutate_params(best, rng: np.random.Generator, step: float):
    out = {
        "logits": np.asarray(best["logits"], dtype=float).copy(),
        "means": np.asarray(best["means"], dtype=float).copy(),
        "log_scales": np.asarray(best["log_scales"], dtype=float).copy(),
    }
    out["logits"] += rng.normal(scale=0.35 * step, size=out["logits"].shape)
    out["means"] += rng.normal(scale=np.array([0.25, 0.30]) * step, size=out["means"].shape)
    out["log_scales"] += rng.normal(scale=0.20 * step, size=out["log_scales"].shape)
    out["means"][:, 0] = np.clip(out["means"][:, 0], -2.0, 4.5)
    out["means"][:, 1] = np.clip(out["means"][:, 1], -2.0, 8.5)
    out["log_scales"] = np.clip(out["log_scales"], np.log(0.25), np.log(4.0))
    return out


def format_priority_function(params):
    logits = np.array2string(np.asarray(params["logits"], dtype=float), precision=6, separator=", ")
    means = np.array2string(np.asarray(params["means"], dtype=float), precision=6, separator=", ")
    log_scales = np.array2string(np.asarray(params["log_scales"], dtype=float), precision=6, separator=", ")
    return f"""def priority(beta: float):
    beta = float(beta)
    logits = np.array({logits}, dtype=float)
    means = np.array({means}, dtype=float)
    log_scales = np.array({log_scales}, dtype=float)

    weights = np.exp(logits - np.max(logits))
    weights = weights / np.sum(weights)
    scales = np.exp(log_scales)

    def sampler(n_samples: int, rng: np.random.Generator) -> np.ndarray:
        n_samples = int(n_samples)
        if n_samples <= 0:
            return np.empty((0, { _DIM }), dtype=float)
        ids = rng.choice(len(weights), size=n_samples, p=weights)
        noise = rng.normal(size=(n_samples, { _DIM }))
        return means[ids] + noise * scales[ids]

    def pdf(x: np.ndarray) -> np.ndarray:
        x = _as_2d(x)
        const = 1.0 / np.sqrt(2.0 * np.pi)
        density = np.zeros(len(x), dtype=float)
        for i in range(len(weights)):
            z = (x - means[i]) / scales[i]
            comp = np.prod(const * np.exp(-0.5 * z * z) / scales[i], axis=1)
            density += weights[i] * comp
        return np.maximum(density, _PDF_FLOOR)

    return sampler, pdf"""


def run_evolution(beta: float = 3.0, target_low: float = 0.94, target_high: float = 0.95, max_iters: int = 2000):
    rng = np.random.default_rng(_BASE_SEED + 71)
    best_params = {
        "logits": np.array([0.0, 1.5, 0.6, -0.4], dtype=float),
        "means": np.array([
            [0.0, 0.0],
            [2.0, 5.4],
            [1.8, 4.8],
            [2.3, 6.0],
        ], dtype=float),
        "log_scales": np.log(np.array([
            [1.0, 1.0],
            [0.7, 0.55],
            [1.0, 0.9],
            [0.9, 0.65],
        ], dtype=float)),
    }

    best_priority = build_priority(best_params)
    best_fitness = evaluate(best_priority, beta)
    log_lines = [f"iter=0 accepted=True fitness={best_fitness:.6f}"]

    for it in range(1, max_iters + 1):
        step = max(0.15, 1.0 - 0.0004 * it)
        cand_params = mutate_params(best_params, rng, step=step)
        cand_priority = build_priority(cand_params)
        cand_fitness = evaluate(cand_priority, beta)
        accepted = cand_fitness > best_fitness

        if accepted:
            best_fitness = cand_fitness
            best_params = cand_params
        log_lines.append(f"iter={it} accepted={accepted} fitness={cand_fitness:.6f} best={best_fitness:.6f}")

        if target_low <= best_fitness <= target_high:
            return {
                "iterations": it,
                "best_fitness": best_fitness,
                "best_params": best_params,
                "log_lines": log_lines,
            }

    return {
        "iterations": max_iters,
        "best_fitness": best_fitness,
        "best_params": best_params,
        "log_lines": log_lines,
    }


def main():
    result = run_evolution(beta=3.0, target_low=0.94, target_high=0.95, max_iters=3000)
    print(f"initial_beta=3.0")
    print(f"evolution_iterations={result['iterations']}")
    print(f"best_fitness={result['best_fitness']:.6f}")
    print("final_priority:\n")
    print(format_priority_function(result["best_params"]))
    print("\nEVOLUTION_LOG_BEGIN")
    for line in result["log_lines"]:
        print(line)
    print("EVOLUTION_LOG_END")


if __name__ == "__main__":
    main()
