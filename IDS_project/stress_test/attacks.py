"""
Attack generation, with and without domain constraints.

The gradient is supplied through a callable so the same attack loops work for
the linear victim now and a differentiable non-linear victim later. For a
LinearSVC the gradient of the decision function is the weight vector, constant
across the input space; for an MLP it varies per sample, but neither attack loop
needs to know which it is holding.
"""

import numpy as np


def linear_gradient_fn(model):
    """Gradient of a linear decision function f(x) = w.x + b, i.e. w."""
    w = model.coef_[0]

    def gradient(X):
        return np.broadcast_to(w, (X.shape[0], w.shape[0]))

    return gradient


def mlp_gradient_fn(model):
    """Gradient of an sklearn MLPClassifier's output logit w.r.t. its input.

    Computed analytically rather than numerically: a forward pass caching the
    pre-activations, then the chain rule backwards through the ReLU masks. The
    logistic output layer is monotonic, so differentiating the logit gives the
    same sign - and therefore the same attack direction - as differentiating the
    probability, without the vanishing factor near saturation.
    """
    Ws, bs = model.coefs_, model.intercepts_

    def gradient(X):
        a = np.asarray(X, dtype=float)
        pre = []
        for i, (W, b) in enumerate(zip(Ws, bs)):
            z = a @ W + b
            pre.append(z)
            a = np.maximum(z, 0) if i < len(Ws) - 1 else z

        g = np.ones((a.shape[0], Ws[-1].shape[1]))
        for i in range(len(Ws) - 1, -1, -1):
            g = g @ Ws[i].T
            if i > 0:
                g = g * (pre[i - 1] > 0)
        return g

    return gradient


def _attackable(model, X, y):
    """Attack samples the victim currently gets right - the only ones worth attacking."""
    y_pred = model.predict(X)
    return (y_pred == 1) & (np.asarray(y) == 1)


def fgsm(model, X, y, epsilon, projector=None, gradient_fn=None, mask=None):
    """Single-step FGSM.

    With projector=None this reproduces the original unconstrained attack. With
    a projector, the perturbed vector is snapped back onto the realizable-flow
    set before being returned.

    Pass `mask` to attack a caller-chosen set of samples. Transfer attacks need
    this: the perturbation is built from a surrogate's gradient, but the samples
    worth perturbing are the ones the *target* currently detects.
    """
    gradient_fn = gradient_fn or linear_gradient_fn(model)
    if mask is None:
        mask = _attackable(model, X, y)

    X_adv = np.array(X, dtype=float, copy=True)
    if not mask.any():
        return X_adv, mask

    grad = gradient_fn(X_adv[mask])
    # Descending the decision function pushes attack samples toward the
    # 'normal' side of the boundary.
    X_adv[mask] -= epsilon * np.sign(grad)

    if projector is not None:
        X_adv[mask] = projector.project(X_adv[mask], X[mask])

    return X_adv, mask


def _iterative(model, X, y, epsilon, steps, alpha, projector, gradient_fn,
               mask, random_start, rng, keep_first_evasion, eval_model=None):
    """Shared loop for BIM and PGD.

    Both take small signed-gradient steps and clip back into the L-infinity ball
    of radius epsilon after each one. The only difference is where they start:
    BIM from the clean sample, PGD from a random point inside the ball.

    On a linear model with no projector both collapse to FGSM - every step walks
    to the same corner of the box, which is already optimal there. They become
    genuinely distinct attacks once a projector is supplied, because each step
    must be re-projected onto the realizable set and the unconstrained optimum
    is no longer reachable.

    keep_first_evasion retains, per sample, the earliest iterate that evaded
    detection rather than the final one. That yields a perturbation close to the
    minimum needed, which makes the mean-perturbation-distance metric
    informative instead of the constant epsilon * sqrt(d) that single-step FGSM
    always produces.

    eval_model decides whether a sample counts as evaded. It differs from
    `model` in transfer attacks, where the gradient comes from a surrogate but
    success is defined by the real target.
    """
    gradient_fn = gradient_fn or linear_gradient_fn(model)
    alpha = alpha if alpha is not None else max(epsilon / 10.0, 1e-8)
    eval_model = eval_model or model
    if mask is None:
        mask = _attackable(model, X, y)

    X_adv_full = np.array(X, dtype=float, copy=True)
    if not mask.any():
        return X_adv_full, mask

    X_clean = np.array(X[mask], dtype=float, copy=True)
    X_work = X_clean.copy()

    if random_start:
        rng = rng if rng is not None else np.random.default_rng(0)
        X_work = X_work + rng.uniform(-epsilon, epsilon, size=X_work.shape)
        if projector is not None:
            X_work = projector.project(X_work, X_clean)

    best = X_clean.copy()
    evaded_already = np.zeros(len(X_clean), dtype=bool)

    for _ in range(steps):
        grad = gradient_fn(X_work)
        X_work = X_work - alpha * np.sign(grad)

        # Stay inside the epsilon budget...
        X_work = np.clip(X_work, X_clean - epsilon, X_clean + epsilon)
        # ...and inside the set of flows that could actually exist.
        if projector is not None:
            X_work = projector.project(X_work, X_clean)

        if keep_first_evasion:
            newly_evaded = (eval_model.predict(X_work) == 0) & ~evaded_already
            if newly_evaded.any():
                best[newly_evaded] = X_work[newly_evaded]
                evaded_already |= newly_evaded
        # Samples that never evade keep the last iterate, which is the
        # strongest attempt made against them.
        best[~evaded_already] = X_work[~evaded_already]

    X_adv_full[mask] = best
    return X_adv_full, mask


def bim(model, X, y, epsilon, steps=40, alpha=None, projector=None,
        gradient_fn=None, mask=None, keep_first_evasion=True, eval_model=None):
    """Basic Iterative Method (Kurakin et al., 2016).

    FGSM applied repeatedly in small steps, clipped back into the epsilon ball
    after each one. Fully deterministic - always starts from the clean sample.
    """
    return _iterative(
        model, X, y, epsilon, steps, alpha, projector, gradient_fn, mask,
        random_start=False, rng=None, keep_first_evasion=keep_first_evasion,
        eval_model=eval_model,
    )


def pgd(model, X, y, epsilon, steps=40, alpha=None, projector=None,
        gradient_fn=None, mask=None, restarts=1, seed=0,
        keep_first_evasion=True, eval_model=None):
    """Projected Gradient Descent (Madry et al., 2017).

    BIM plus a random start inside the epsilon ball, optionally repeated from
    several starting points. The randomisation is the whole point: one
    deterministic descent can stall against a flat or awkward region of the
    boundary, and restarting elsewhere finds evasions BIM misses. Expect
    PGD >= BIM, with the gap widening once constraints complicate the search.
    """
    rng = np.random.default_rng(seed)
    target = eval_model or model
    best_adv, out_mask, evaded_any = None, None, None

    for _ in range(max(1, restarts)):
        adv, m = _iterative(
            model, X, y, epsilon, steps, alpha, projector, gradient_fn, mask,
            random_start=True, rng=rng, keep_first_evasion=keep_first_evasion,
            eval_model=eval_model,
        )
        if best_adv is None:
            best_adv, out_mask = adv, m
            evaded_any = (target.predict(adv) == 0) & m
            continue
        # Keep whichever restart actually succeeded, per sample.
        newly = (target.predict(adv) == 0) & m & ~evaded_any
        if newly.any():
            best_adv[newly] = adv[newly]
            evaded_any |= newly

    return best_adv, out_mask


def minimal_epsilon_search(model, X, y, projector=None, gradient_fn=None,
                           lo=0.0, hi=2.0, iterations=12):
    """Per-sample binary search for the smallest epsilon that flips the label.

    Returns the epsilon found for each attackable sample (inf where no epsilon
    in [lo, hi] succeeded). This is the honest way to report attack cost: how
    much distortion the evasion actually required, rather than how much budget
    was made available.
    """
    gradient_fn = gradient_fn or linear_gradient_fn(model)
    mask = _attackable(model, X, y)
    n = int(mask.sum())
    if n == 0:
        return np.array([]), mask

    X_clean = np.array(X[mask], dtype=float, copy=True)
    lo_arr = np.full(n, lo, dtype=float)
    hi_arr = np.full(n, hi, dtype=float)

    def evades_at(eps_arr):
        grad = gradient_fn(X_clean)
        cand = X_clean - eps_arr[:, None] * np.sign(grad)
        if projector is not None:
            cand = projector.project(cand, X_clean)
        return model.predict(cand) == 0

    reachable = evades_at(hi_arr)

    for _ in range(iterations):
        mid = (lo_arr + hi_arr) / 2.0
        ok = evades_at(mid)
        hi_arr = np.where(ok, mid, hi_arr)
        lo_arr = np.where(ok, lo_arr, mid)

    result = np.where(reachable, hi_arr, np.inf)
    return result, mask
