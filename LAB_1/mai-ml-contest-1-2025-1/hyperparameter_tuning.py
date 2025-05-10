import itertools

import numpy as np

from cross_validation import kfold_split, cross_val_score_manual


def grid_search_cv(model_factory, param_grid, X, y,
                   cv=None, scoring=None, verbose=False):
    if cv is None:
        cv = kfold_split(X, n_splits=5, shuffle=True, random_state=42)

    keys, values = zip(*param_grid.items()) if param_grid else ([], [])
    all_param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if keys else [{}]

    results = []
    best_score = None
    best_params = None

    for idx, params in enumerate(all_param_combinations):
        model = model_factory(**params)
        scores = cross_val_score_manual(model, X, y, cv=cv, scoring=scoring)
        mean_score = np.mean(scores)
        results.append({
            'params': params,
            'scores': scores,
            'mean_score': mean_score
        })
        if verbose:
            print(f"[{idx+1}/{len(all_param_combinations)}] params={params}  mean_score={mean_score:.4f}")

        if best_score is None or mean_score < best_score:
            best_score = mean_score
            best_params = params.copy()

    return best_params, best_score, results


def random_search_cv(model_factory, param_distributions, X, y,
                     n_iter=20, cv=None, scoring=None, random_state=None, verbose=False):
    rng = np.random.RandomState(random_state)
    if cv is None:
        cv = kfold_split(X, n_splits=5, shuffle=True, random_state=42)

    def sample_value(dist):
        if callable(dist):
            return dist(rng)
        elif isinstance(dist, (list, tuple, np.ndarray)):
            return rng.choice(dist)
        else:
            return dist

    results = []
    best_score = None
    best_params = None

    for i in range(n_iter):
        params = {k: sample_value(v) for k, v in param_distributions.items()}
        model = model_factory(**params)
        scores = cross_val_score_manual(model, X, y, cv=cv, scoring=scoring)
        mean_score = np.mean(scores)
        results.append({'params': params, 'scores': scores, 'mean_score': mean_score})
        if verbose:
            print(f"Iter {i+1}/{n_iter} params={params}  mean_score={mean_score:.4f}")
        if best_score is None or mean_score < best_score:
            best_score = mean_score
            best_params = params.copy()

    return best_params, best_score, results


def ensemble_search_cv(model_factory, tuner_configs, X, y,
                       cv=None, scoring=None, verbose=False):
    if cv is None:
        cv = kfold_split(X, n_splits=5, shuffle=True, random_state=42)

    all_results = {}
    best_score = None
    best_params = None
    best_method = None

    for cfg in tuner_configs:
        method = cfg.get('method')
        if method == 'grid':
            bp, bs, res = grid_search_cv(
                model_factory,
                cfg.get('param_grid', {}),
                X, y,
                cv=cv,
                scoring=scoring,
                verbose=verbose
            )
        elif method == 'random':
            bp, bs, res = random_search_cv(
                model_factory,
                cfg.get('param_distributions', {}),
                X, y,
                n_iter=cfg.get('n_iter', 20),
                cv=cv,
                scoring=scoring,
                random_state=cfg.get('random_state', None),
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown tuner method: {method}")

        all_results[method] = {'best_params': bp, 'best_score': bs, 'results': res}

        if best_score is None or bs < best_score:
            best_score = bs
            best_params = bp
            best_method = method

    return best_method, best_params, best_score, all_results
