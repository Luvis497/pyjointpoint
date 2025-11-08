# -*- coding: utf-8 -*-
# -------------------------------

# @product：PyCharm
# @project：pyjointpoint

# -------------------------------

# @filename：joinpoint_regression.py
# @teim：2025/11/8 14:08
# @name：ShuaiFu Lu
# @email：2301110293@pku.edu.cn

# -------------------------------

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_acovf
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, sqrtm, inv
import warnings
import time
from joblib import Parallel, delayed # <-- 新增导入

# Suppress a common warning from statsmodels when using GLS
warnings.filterwarnings("ignore", "unobserved_slice_variance", UserWarning)

def _fit_single_model(x, y, joinpoints, weights=None, sigma=None, rho=None, ar_maxiter=3):
    """
    Internal function to fit a single joinpoint model for given joinpoints.
    Uses OLS, WLS, or GLS based on inputs.
    """
    # Design matrix
    T = np.ones((len(x), 2 + len(joinpoints)))
    T[:, 1] = x
    for i, jp in enumerate(joinpoints):
        T[:, i + 2] = np.maximum(0, x - jp)

    if rho is not None:
        model = sm.GLSAR(y, T, rho=rho)
    elif sigma is not None:
        model = sm.GLS(y, T, sigma=sigma)
    elif weights is not None:
        model = sm.WLS(y, T, weights=weights)
    else:
        model = sm.OLS(y, T)

    try:
        if rho is not None:
            results = model.iterative_fit(maxiter=ar_maxiter)
        else:
            results = model.fit()
        return results
    except np.linalg.LinAlgError:
        return None


def _evaluate_combination(jp_indices, x, y, weights, sigma, rho, ar_maxiter, min_points_between_jp):
    """
    Fits a model for a single combination of joinpoints and returns its results.
    This function is designed to be called in parallel.
    """
    if len(jp_indices) > 1 and np.any(np.diff(sorted(jp_indices)) < min_points_between_jp):
        return {'sse': np.inf, 'model': None, 'joinpoints': []}

    joinpoint_combination = x[list(jp_indices)] if len(jp_indices) > 0 else []
    model = _fit_single_model(x, y, joinpoint_combination, weights=weights, sigma=sigma, rho=rho, ar_maxiter=ar_maxiter)

    if model:
        return {
            'sse': model.ssr,
            'model': model,
            'joinpoints': sorted(joinpoint_combination)
        }
    else:
        return {'sse': np.inf, 'model': None, 'joinpoints': []}


def _find_best_model_for_k(k, x, y, weights, sigma, rho, min_points_between_jp, ar_maxiter, parallel=False, n_jobs=-1):
    """
    Refactored grid search logic to find the best model for a given number of joinpoints 'k'.
    Can run in parallel.
    """
    n = len(x)
    if k == 0:
        # No combinations to check, just fit the base model
        model = _fit_single_model(x, y, [], weights=weights, sigma=sigma,rho=rho,ar_maxiter=ar_maxiter)
        if model:
            return {
                'sse': model.ssr,
                'model': model,
                'joinpoints': []
            }
        return None

    possible_jp_indices = np.arange(min_points_between_jp, n - min_points_between_jp)
    if len(possible_jp_indices) < k:
        return None  # Not enough points for k joinpoints

    jp_indices_combinations = list(combinations(possible_jp_indices, k))

    if not parallel:

        best_sse_for_k = np.inf
        best_model_for_k = None
        best_jps_for_k = []

        for jp_indices in jp_indices_combinations:
            if k > 1 and np.any(np.diff(sorted(jp_indices)) < min_points_between_jp):
                continue

            res = _evaluate_combination(jp_indices, x, y, weights, sigma, rho, ar_maxiter, min_points_between_jp)

            if res['sse'] < best_sse_for_k:
                best_sse_for_k = res['sse']
                best_model_for_k = res['model']
                best_jps_for_k = res['joinpoints']

        if best_model_for_k:
            return {
                'sse': best_sse_for_k,
                'model': best_model_for_k,
                'joinpoints': best_jps_for_k
            }
        return None

    else:

        #  joblib  _evaluate_combination
        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_combination)(jp_indices, x, y, weights, sigma, rho, ar_maxiter, min_points_between_jp)
            for jp_indices in jp_indices_combinations
        )


        if not results:
            return None


        valid_results = [res for res in results if res and res['model'] is not None]
        if not valid_results:
            return None

        best_result = min(valid_results, key=lambda item: item['sse'])
        return best_result


# --- AR matrix ---
def _ar_acovf_matrix(phi, n):
    """
    Compute the autocovariance matrix of an AR(p) model using statsmodels.
    """
    ar_poly = np.concatenate(([1], -np.array(phi)))
    ma_poly = [1]
    acovf = arma_acovf(ar=ar_poly, ma=ma_poly, nobs=n)
    acovf = acovf / acovf.max()
    autocovariance_matrix = toeplitz(acovf)
    return autocovariance_matrix


# --- Model Selection Criteria Functions  ---

def _calculate_bic(sse, n, k):
    if sse <= 0: return np.inf
    penalty_coeff = 2 * (k + 1)
    return np.log(sse / n) + penalty_coeff * np.log(n) / n

def _calculate_bic3(sse, n, k):
    if sse <= 0: return np.inf
    penalty_coeff = 3 * k + 2
    return np.log(sse / n) + penalty_coeff * np.log(n) / n

def _project_matrix(X):
    try:
        return X @ np.linalg.pinv(X.T @ X) @ X.T
    except np.linalg.LinAlgError:
        return np.zeros_like(X)

def _calculate_partial_r_squared(y, X_base, z_new):
    try:
        H_base = _project_matrix(X_base)
        I_H = np.eye(len(y)) - H_base
        num = (y.T @ I_H @ z_new) ** 2
        den_y = y.T @ I_H @ y
        den_z = z_new.T @ I_H @ z_new
        if den_y < 1e-9 or den_z < 1e-9: return 0.0
        r_sq = num / (den_y * den_z)
        return r_sq
    except np.linalg.LinAlgError: return 0.0

def _calculate_wbic(sse, n, k, x, y, joinpoints):
    if k == 0: return _calculate_bic(sse, n, k)
    if sse <= 0: return np.inf
    r_squared_max = 0.0
    if k > 0:
        sorted_jps = sorted(joinpoints)
        partial_r_squares = []
        for s in range(1, k + 1):
            X_base = np.ones((n, 2 + (s - 1))) if s > 1 else np.ones((n, 2))
            if s > 1:
                X_base[:, 1] = x
                for j in range(s - 1):
                    X_base[:, j + 2] = np.maximum(0, x - sorted_jps[j])
            else:
                X_base[:, 1] = x
            z_s = np.maximum(0, x - sorted_jps[s - 1])
            pr2 = _calculate_partial_r_squared(y, X_base, z_s)
            partial_r_squares.append(pr2)
        if partial_r_squares:
            r_squared_max = np.max(partial_r_squares)
    penalty_coeff = k * (2 + r_squared_max)
    return np.log(sse / n) + penalty_coeff * np.log(n) / n

def _run_permutation_test(k_null, k_alt, x, y, weights, sigma,rho, min_points_between_jp, num_permutations,ar_maxiter, parallel, n_jobs):
    """
    Performs a permutation test. Note: The permutation loop itself is not parallelized here,
    but the model fitting within each permutation WILL be if parallel=True.
    """
    n = len(x)

    # 1. Fit models on original data to get observed F-statistic
    null_model_res = _find_best_model_for_k(k_null, x, y, weights, sigma, rho, min_points_between_jp, ar_maxiter, parallel, n_jobs)
    alt_model_res = _find_best_model_for_k(k_alt, x, y, weights, sigma, rho, min_points_between_jp, ar_maxiter, parallel, n_jobs)

    if not null_model_res or not alt_model_res:
        print("Warning: Could not fit model for permutation test. Skipping.")
        return 1.0

    sse_null, model_null = null_model_res['sse'], null_model_res['model']
    sse_alt = alt_model_res['sse']

    if sse_alt >= sse_null: return 1.0

    stat_obs = sse_null - sse_alt
    residuals = model_null.resid.copy()
    fitted_values = model_null.fittedvalues

    # 3. Prepare for permutation
    if weights is not None:
        sqrt_weights = np.sqrt(weights)
        scaled_res = residuals * sqrt_weights
        scale_back = 1 / sqrt_weights
        use_gls = False
    elif (sigma is not None) or (rho is not None):
        if (rho is not None):
            phi=model_null.rho
            sigma=_ar_acovf_matrix(phi,n)
        try:
            V_inv_sqrt = sqrtm(np.linalg.inv(sigma))
            scaled_res = V_inv_sqrt @ residuals
            scale_back_matrix = sqrtm(sigma)
            use_gls = True
        except:
            print("Warning: GLS permutation fallback to unscaled (approximation).")
            scaled_res = residuals.copy()
            scale_back_matrix = np.eye(n)
            use_gls = False
    else:
        scaled_res = residuals.copy()
        scale_back = np.ones(n)
        use_gls = False

    # 4. Permutation loop
    extreme_count = 0
    for i in range(num_permutations):
        if (i + 1) % 100 == 0:
            print(f"    ... running permutation {i+1}/{num_permutations}")

        np.random.shuffle(scaled_res)
        if use_gls:
            y_perm = fitted_values + (scale_back_matrix @ scaled_res)
        else:
            y_perm = fitted_values + (scaled_res * scale_back)

        perm_null_res = _find_best_model_for_k(k_null, x, y_perm, weights, sigma, rho, min_points_between_jp, ar_maxiter, parallel, n_jobs)
        perm_alt_res = _find_best_model_for_k(k_alt, x, y_perm, weights, sigma, rho, min_points_between_jp, ar_maxiter, parallel, n_jobs)

        if not perm_null_res or not perm_alt_res: continue

        sse_null_perm, sse_alt_perm = perm_null_res['sse'], perm_alt_res['sse']
        if sse_alt_perm >= sse_null_perm: continue

        stat_perm = sse_null_perm - sse_alt_perm
        if stat_perm >= stat_obs:
            extreme_count += 1

    p_value = (extreme_count + 1) / (num_permutations + 1)
    return p_value

# --- joinpoint regression ---
def joinpoint_regression(x, y, type='linear', max_joinpoints=3,
                         model_selection_method='BIC',
                         heteroscedasticity_option='constant',
                         error_model_fit=None,
                         rho=None,
                         se=None, vcm=None,
                         min_points_between_jp=2,
                         permutation_alpha=0.05,
                         ar_maxiter=10,
                         num_permutations=1999,
                         parallel=False, n_jobs=-1): # <-- 新增参数
    """
    Performs joinpoint regression to identify significant changes in trend.

    Args:
        x (array-like): Independent variable, typically time (e.g., year). Must be sorted.
        y (array-like): Dependent variable (e.g., rates).
        type (str): 'log-linear' for rates (y=log(rate)), 'linear' for other data.
        max_joinpoints (int): The maximum number of joinpoints to consider.
        model_selection_method (str): 'BIC', 'BIC3', 'WBIC', 'Permutation'.
        heteroscedasticity_option (str): 'constant', 'ar1', 'se', 'pv', 'vcm'.
        error_model_fit (str): 'u' (uncorrelated), 'ar' (autoregressive).
        rho (int): AR order for 'ar' model.
        se (array-like, optional): Standard errors for 'se' option.
        vcm (array-like, optional): Variance-covariance matrix for 'vcm' option.
        min_points_between_jp (int): Min data points between joinpoints.
        permutation_alpha (float): Alpha level for permutation test.
        num_permutations (int): Number of permutations.
        ar_maxiter (int): Max iterations for AR model fitting.
        parallel (bool): If True, use parallel processing for the grid search.
        n_jobs (int): Number of CPU cores to use for parallel processing.
                      -1 means use all available cores. Defaults to -1.
    Returns:
        dict: A dictionary containing the results.
    """
    x = np.array(x); y = np.array(y); n = len(x)
    if len(x) != len(y): raise ValueError("x and y must have the same length.")

    if type=='log-linear':
        y=np.log(y)
    elif type=='linear':
        pass
    else:
        raise TypeError(f"Unknown type {type} in joinpoint regression. use 'log-linear' or 'linear'")

    weights, sigma = None, None
    option_lower = heteroscedasticity_option.lower()

    if option_lower in ['se', 'standard error']:
        if se is None: raise ValueError("Standard errors 'se' must be provided.")
        if type == 'log-linear': weights = np.exp(y)**2 / (np.array(se)**2)
        else: weights = 1 / (np.array(se)**2)
    elif option_lower in ['pv', 'poisson variance']:
        y_for_weights = np.exp(y) if type == 'log-linear' else y
        weights = 1 / np.where(y_for_weights > 0, y_for_weights, 0.5)
        if type == 'log-linear': weights = y_for_weights # For log-linear, weights are E(y) not 1/E(y)
    elif option_lower in ['vcm', 'variance-covariance matrix']:
        if vcm is None: raise ValueError("Variance-covariance matrix 'vcm' must be provided.")
        sigma = np.array(vcm)

    # Autoregressive error handling
    if error_model_fit in ['ar'] and isinstance(rho, int):
        if vcm is None:
            warnings.warn("Variance-covariance matrix 'vcm' is provided. But Method GLSAR in statsmodels does not support both pre-specified variance(heteroscedasticity in WLS/GLS) and AR errors(correlated error) simultaneously. The AR model will be prioritized. correlated error and heteroscedasticity. The AR model will be prioritized. More complex method likes PanelOLS(in linearmodels(https://bashtage.github.io/linearmodels)) don't available now.", UserWarning)
            sigma, weights = None, None # Prioritize AR model
        if se is None:
            warnings.warn("Standard errors 'se' is provided. But Method GLSAR in statsmodels does not support both pre-specified variance(heteroscedasticity in WLS/GLS) and AR errors(correlated error) simultaneously. The AR model will be prioritized. correlated error and heteroscedasticity. The AR model will be prioritized. More complex method likes PanelOLS(in linearmodels(https://bashtage.github.io/linearmodels)) don't available now.", UserWarning)
            sigma, weights = None, None # Prioritize AR model

    else:
        rho = None


    if parallel:
        if n_jobs == -1:
            import os
            num_cores = os.cpu_count()
            print(f"Finding best model using all available {num_cores} cores...")
        else:
            print(f"Finding best model using {n_jobs} parallel jobs...")


    model_summaries = []
    for k in range(max_joinpoints + 1):
        # 将并行参数传递给核心搜索函数
        best_model_info = _find_best_model_for_k(k, x, y, weights, sigma, rho, min_points_between_jp, ar_maxiter, parallel, n_jobs)
        if best_model_info:
            summary = { 'k': k, **best_model_info }
            summary['bic'] = _calculate_bic(summary['sse'], n, k)
            summary['bic3'] = _calculate_bic3(summary['sse'], n, k)
            summary['wbic'] = _calculate_wbic(summary['sse'], n, k, x, y, summary['joinpoints'])
            model_summaries.append(summary)

    if not model_summaries: raise RuntimeError("Could not fit any models.")

    model_summary_df = pd.DataFrame(model_summaries).set_index('k')

    selection_lower = model_selection_method.lower()
    if selection_lower in ['bic', 'bic3', 'wbic', 'modified bic']:
        crit = 'bic3' if selection_lower == 'modified bic' else selection_lower
        best_k = model_summary_df[crit].idxmin()
    elif selection_lower in ['permutation test', 'permutation']:
        print("\nInfo: Starting sequential permutation testing procedure.")
        best_k = 0
        for k_test in range(max_joinpoints):
            start_time_perm = time.time()
            # Bonferroni correction
            corrected_alpha = permutation_alpha / (max_joinpoints - k_test)
            print(f"\n--- Testing H0: k={k_test} vs H1: k={k_test + 1} (Corrected alpha={corrected_alpha:.4f}) ---")
            # Pass parallel options to permutation test
            p_value = _run_permutation_test(k_test, k_test + 1, x, y, weights, sigma, rho, min_points_between_jp, num_permutations,ar_maxiter, parallel, n_jobs)
            end_time_perm = time.time()
            print(f"--- Test complete in {end_time_perm - start_time_perm:.2f} seconds. ---")
            if p_value < corrected_alpha:
                print(f"Result: p={p_value:.4f} < {corrected_alpha:.4f}. Reject H0. The model has at least {k_test + 1} joinpoints.")
                best_k = k_test + 1
            else:
                print(f"Result: p={p_value:.4f} >= {corrected_alpha:.4f}. Fail to reject H0. Selecting k={k_test} as the final model.")
                break
        print(f"\nPermutation testing concluded. Final selected number of joinpoints: k = {best_k}")
    else:
        raise ValueError(f"Model selection method '{model_selection_method}' is not implemented.")

    best_model_row = model_summary_df.loc[best_k]
    best_joinpoints = best_model_row['joinpoints'] if best_k > 0 else []
    final_model = best_model_row['model']

    params = final_model.params
    # The first slope is params[1]. Subsequent params are changes in slope.
    slope_changes = params[2:]
    slopes = np.cumsum(np.insert(slope_changes, 0, params[1]))

    apcs = 100 * (np.exp(slopes) - 1) if type == 'log-linear' else slopes * 100
    apc_label = 'APC (%)' if type == 'log-linear' else 'Annual Change'

    segment_starts = np.concatenate(([x[0]], best_joinpoints)) if best_k > 0 else [x[0]]
    segment_ends = np.concatenate((best_joinpoints, [x[-1]])) if best_k > 0 else [x[-1]]
    apc_data = [{'Segment': i + 1, 'Start Year': s, 'End Year': e, 'Slope (b)': b, apc_label: a}
                for i, (s, e, b, a) in enumerate(zip(segment_starts, segment_ends, slopes, apcs))]

    return {
        'best_k': best_k, 'best_joinpoints': best_joinpoints,
        'model_summary_df': model_summary_df.reset_index()[['k', 'joinpoints', 'sse', 'bic', 'bic3', 'wbic']],
        'apc_summary_df': pd.DataFrame(apc_data), 'fitted_model': final_model,
        'fitted_values': final_model.fittedvalues, 'input_type': type
    }



if __name__ == '__main__':
    # --- 1. 生成模拟数据 ---
    # 创建一个包含两个真实连接点的时间序列数据
    np.random.seed(42)
    x_years = np.arange(1980, 2021) # 41个数据点

    # 真实连接点在 1995 和 2010
    jp1, jp2 = 1995, 2010

    # 分段定义真实的 log(rate)
    y_log_true = np.zeros_like(x_years, dtype=float)

    # 第一段: 1980-1994 (斜率 0.03, 对应 APC ~3%)
    mask1 = x_years < jp1
    y_log_true[mask1] = -4.0 + 0.03 * (x_years[mask1] - 1980)

    # 第二段: 1995-2009 (斜率 -0.02, 对应 APC ~-2%)
    mask2 = (x_years >= jp1) & (x_years < jp2)
    y_log_true[mask2] = y_log_true[mask1][-1] - 0.02 * (x_years[mask2] - x_years[mask1][-1])

    # 第三段: 2010-2020 (斜率 0.01, 对应 APC ~1%)
    mask3 = x_years >= jp2
    y_log_true[mask3] = y_log_true[mask2][-1] + 0.01 * (x_years[mask3] - x_years[mask2][-1])

    # 添加一些噪音
    y_log_observed = y_log_true + np.random.normal(0, 0.05, size=len(x_years))

    # 假设我们分析的是比率，所以使用 log-linear 模型
    rates = np.exp(y_log_observed)

    # --- 2. 设置连接点回归参数 ---
    # 为了突出性能差异，我们选择一个相对较大的 max_joinpoints
    MAX_JP = 4
    MIN_PTS = 3 # 在连接点之间至少需要3个点

    print("="*60)
    print("Joinpoint Regression Performance Comparison")
    print(f"Dataset Size: {len(x_years)} points")
    print(f"Max Joinpoints to test: {MAX_JP}")
    print("="*60)

    # --- 3. 运行串行版本 ---
    print("\n--- Running in Serial Mode (parallel=False) ---")
    start_time_serial = time.time()

    results_serial = joinpoint_regression(
        x=x_years,
        y=rates, # 传入原始比率
        type='log-linear', # 指定模型类型
        max_joinpoints=MAX_JP,
        min_points_between_jp=MIN_PTS,
        model_selection_method='BIC',
        parallel=False # 禁用并行
    )

    end_time_serial = time.time()
    duration_serial = end_time_serial - start_time_serial

    print("\n--- Serial Execution Finished ---")
    print(f"Time Taken: {duration_serial:.2f} seconds")
    print(f"Best number of joinpoints (k): {results_serial['best_k']}")
    print(f"Identified joinpoints: {results_serial['best_joinpoints']}")
    print("\nModel Selection Summary (Serial):")
    print(results_serial['model_summary_df'])


    # --- 4. 运行并行版本 ---
    print("\n" + "="*60)
    print("\n--- Running in Parallel Mode (parallel=True) ---")
    start_time_parallel = time.time()

    results_parallel = joinpoint_regression(
        x=x_years,
        y=rates,
        type='log-linear',
        max_joinpoints=MAX_JP,
        min_points_between_jp=MIN_PTS,
        model_selection_method='BIC',
        parallel=True, # 启用并行
        n_jobs=-1      # 使用所有可用的CPU核心
    )

    end_time_parallel = time.time()
    duration_parallel = end_time_parallel - start_time_parallel

    print("\n--- Parallel Execution Finished ---")
    print(f"Time Taken: {duration_parallel:.2f} seconds")
    print(f"Best number of joinpoints (k): {results_parallel['best_k']}")
    print(f"Identified joinpoints: {results_parallel['best_joinpoints']}")
    print("\nModel Selection Summary (Parallel):")
    print(results_parallel['model_summary_df'])

    # --- 5. 比较结果和性能 ---
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print(f"Serial execution time:   {duration_serial:.4f} seconds")
    print(f"Parallel execution time: {duration_parallel:.4f} seconds")
    if duration_serial > 0 and duration_parallel > 0:
        speedup = duration_serial / duration_parallel
        print(f"Speedup factor:          {speedup:.2f}x")
    print("="*60)

    # 验证结果是否一致
    assert results_serial['best_k'] == results_parallel['best_k'], "Best k differs between serial and parallel runs!"
    assert results_serial['best_joinpoints'] == results_parallel['best_joinpoints'], "Best joinpoints differ!"
    print("\n✅ Verification successful: Serial and parallel runs produced identical results.")

    # --- 6. 可视化最终结果 ---
    final_results = results_parallel # 使用并行或串行结果均可

    plt.figure(figsize=(12, 7))
    plt.plot(x_years, y_log_observed, 'o', label='Observed log(Rate)', alpha=0.6)

    # 绘制拟合的连接点模型
    if final_results['input_type'] == 'log-linear':
        plt.plot(x_years, final_results['fitted_values'], 'r-', linewidth=2, label=f"Fitted Joinpoint Model (k={final_results['best_k']})")
    else: # if linear
         plt.plot(x_years, np.log(final_results['fitted_values']), 'r-', linewidth=2, label=f"Fitted Joinpoint Model (k={final_results['best_k']})")

    # 标记连接点
    for jp in final_results['best_joinpoints']:
        plt.axvline(x=jp, color='g', linestyle='--', label=f'Joinpoint at {jp}')

    plt.title('Joinpoint Regression Results')
    plt.xlabel('Year')
    plt.ylabel('log(Rate)')
    # 合并图例中的重复标签
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    print("\nFinal APC Summary:")
    print(final_results['apc_summary_df'].to_string(index=False))
