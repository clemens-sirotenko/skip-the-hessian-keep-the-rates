import math
import numpy as np
import scipy
from scipy.special import logsumexp, softmax
from datetime import datetime
from collections import defaultdict
import warnings


"""
The BaseSmoothOracle and OracleCallsCounter classes are originally found in
Nikita Doikov's super-newton repository: https://github.com/doikov/super-newton
which has the following MIT License:

MIT License

Copyright (c) 2022 Nikita

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

leapssn is a modification of the original Doikov's super_newton algorithm.

"""


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of the function at point x.
        """
        raise NotImplementedError("Func oracle is not implemented.")

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError("Grad oracle is not implemented.")

    def grad_i(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError("Grad oracle is not implemented.")

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError("Hessian oracle is not implemented.")

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)

    def third_vec_vec(self, x, v):
        """
        Computes tensor-vector-vector product with the third derivative tensor
        D^3 f(x)[v, v].
        """
        raise NotImplementedError("Third derivative oracle is not implemented.")


class OracleCallsCounter(BaseSmoothOracle):
    """
    Wrapper to count oracle calls.
    """

    def __init__(self, oracle):
        self.oracle = oracle
        self.func_calls = 0
        self.grad_calls = 0
        self.grad_i_calls = 0
        self.hess_calls = 0
        self.hess_vec_calls = 0
        self.third_vec_vec_calls = 0
        self.func_at_hess_eval = []

    def func(self, x):
        self.func_calls += 1
        return self.oracle.func(x)

    def grad(self, x):
        self.grad_calls += 1
        return self.oracle.grad(x)
    def grad_i(self, x, i):
        self.grad_i_calls += 1
        return self.oracle.grad_i(x, i)
    def hess(self, x):
        self.hess_calls += 1
        self.func_at_hess_eval.append(self.oracle.func(x))
        return self.oracle.hess(x)

    def hess_vec(self, x, v):
        self.hess_vec_calls += 1
        return self.oracle.hess_vec(x, v)

    def third_vec_vec(self, x, v):
        self.third_vec_vec_calls += 1
        return self.oracle.third_vec_vec(x, v)


class OptimizationMethods(BaseSmoothOracle):

    
    @staticmethod
    def cubic_newton_step(g, A, H, B=None, eps=1e-8):
        """
        Computes minimizer of the following function:
        f(x) = <g, x> + 1/2 * <Ax, x> + H/3 * ||x||^3,
        by finding the root of the equation:
        h(r) = r - ||(A + HrB)^{-1} g|| = 0.
        """
        n = g.shape[0]
        if B is None:
            B = np.eye(n)
            l2_norm_sqr = lambda x: x.dot(x)
        else:
            l2_norm_sqr = lambda x: B.dot(x).dot(x)

        def f(T, T_norm):
            return g.dot(T) + 0.5 * A.dot(T).dot(T) + H * T_norm ** 3 / 3.0

        def h(r, der=False):
            ArB_cho_factor = scipy.linalg.cho_factor(A + H * r * B, lower=False)
            T = scipy.linalg.cho_solve(ArB_cho_factor, -g)
            T_norm = l2_norm_sqr(T) ** 0.5
            h_r = r - T_norm
            if der:
                BT = B.dot(T)
                h_r_prime = 1 + H / T_norm * \
                            scipy.linalg.cho_solve(ArB_cho_factor, BT).dot(BT)
            else:
                h_r_prime = None
            return h_r, T_norm, T, h_r_prime

        try:
            max_r = 1.0
            max_iters = 50
            # Find max_r such that h(max_r) is nonnegative
            for i in range(max_iters):
                h_r, T_norm, T, _ = h(max_r)
                if h_r < -eps:
                    max_r *= 2
                elif -eps <= h_r <= eps:
                    return T, f(T, T_norm), "success"
                else:
                    break

            # Univariate Newton's
            r = max_r
            for i in range(max_iters):
                h_r, T_norm, T, h_r_prime = h(r, der=True)
                if -eps <= h_r <= eps:
                    return T, f(T, T_norm), "success"
                r -= h_r / h_r_prime

        except (np.linalg.LinAlgError, ValueError) as e:
            return np.zeros(n), 0.0, "linalg_error"

        return np.zeros(n), 0.0, "iterations_exceeded"
    def cubic_newton(oracle, x_0, n_iters=100, H_0=1.0, adaptive_search=True,
                    trace=True, B=None, eps=1e-8, H_min=1e-5, f_star=None,
                    warnings=False):
        

        


        """
        Run Cubic Newton Method 
        for 'n_iters' iterations, minimizing smooth function.

        'oracle' is an instance of BaseSmoothOracle representing the objective.
        """
        oracle = OracleCallsCounter(oracle)
        start_timestamp = datetime.now()

        # Initialization of the Euclidean metric
        if B is None:
            l2_norm_sqr = lambda x: x.dot(x)
            dual_norm_sqr = lambda x: x.dot(x)
            to_dual = lambda x: x
        else:
            l2_norm_sqr = lambda x: B.dot(x).dot(x)
            to_dual = lambda x: B.dot(x)
            if Binv is None:
                Binv = np.linalg.inv(B)
            dual_norm_sqr = lambda x: Binv.dot(x).dot(x)
        # if Binv is None:
        precond = lambda g: g
        # else:
        #     precond = lambda g: Binv.dot(g)

        # Initialization of the method
        x_k = np.copy(x_0)
        f_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)
        grad_k_norm_sqr = dual_norm_sqr(grad_k)
        g_k_norm = np.sqrt(grad_k_norm_sqr)

        H_k = H_0

        history = defaultdict(list) if trace else None
        status = ""

        # Main loop
        for k in range(n_iters + 1):

            if trace:
                history['func'].append(f_k)
                history['grad_calls'].append(oracle.grad_calls)
                history["grad_sqr_norm"].append(grad_k_norm_sqr)
                history["grad_norm"].append(g_k_norm)
                history['H_k'].append(H_k)
                history['hess_calls'].append(oracle.hess_calls)
                history['new'].append(oracle.func_at_hess_eval)
                history['time'].append(
                    (datetime.now() - start_timestamp).total_seconds())

            if f_star is not None and f_k - f_star < eps:
                status = "success, %d iters" % k
                break

            if k == n_iters:
                status = "iterations_exceeded"
                break

            g_k = oracle.grad(x_k)
            Hess_k = oracle.hess(x_k)
            history["func_at_hess_eval"].append(oracle.func(x_k))
            adaptive_search_max_iter = 100
            for i in range(adaptive_search_max_iter + 1):
                if i == adaptive_search_max_iter:
                    if warnings:
                        print('W: adaptive_iterations_exceeded', flush=True)
                    break

                # Compute the Cubic Newton step
                x_delta, model_value, message = OptimizationMethods.cubic_newton_step(g_k, Hess_k,
                                                                0.5 * H_k, B, 1e-8)
                if message != "success" and warnings:
                    print('W: cubic_newton_step: %s' % message, end=' ', flush=True)
                f_new = oracle.func(x_k + x_delta)
                
                grad_new = oracle.grad(x_k + x_delta)
                grad_new_norm_sqr = dual_norm_sqr(grad_new)
                history["func_at_lssolve"].append(f_new)
                history["grad_norm_at_lssolve"].append(grad_new_norm_sqr**0.5)

                if not adaptive_search:
                    break

                # Check condition for H_k
                if f_new <= f_k + model_value:
                    H_k *= 0.5
                    H_k = max(H_k, H_min)
                    break

                H_k *= 2

            # Update the point
            x_k += x_delta
            f_k = f_new
            grad_k = oracle.grad(x_k)
            grad_k_norm_sqr = dual_norm_sqr(grad_k)
            g_k_norm = np.sqrt(grad_k_norm_sqr)
        return x_k, status, history





    def gradient_method(
        oracle,
        x_0,
        max_iter=1000,
        L_0=1.0,
        line_search=True,
        trace=True,
        B=None,
        Binv=None,
        L_min=1e-5,
        warnings=False,
    ):
        """
        Run the Gradient Method for 'n_iters' iterations,
        minimizing smooth function.

        'oracle' is an instance of BaseSmoothOracle representing the objective.
        """
        oracle = OracleCallsCounter(oracle)
        start_timestamp = datetime.now()

        # Initialization of the Euclidean metric
        if B is None:
            l2_norm_sqr = lambda x: x.dot(x)
            dual_norm_sqr = lambda x: x.dot(x)
            to_dual = lambda x: x
        else:
            l2_norm_sqr = lambda x: B.dot(x).dot(x)
            to_dual = lambda x: B.dot(x)
            if Binv is None:
                Binv = np.linalg.inv(B)
            dual_norm_sqr = lambda x: Binv.dot(x).dot(x)
        if Binv is None:
            precond = lambda g: g
        else:
            precond = lambda g: Binv.dot(g)

        # Initialization of the method
        x_k = np.copy(x_0)
        func_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)
        grad_k_norm_sqr = dual_norm_sqr(grad_k)
        g_k_norm = np.sqrt(grad_k_norm_sqr)
        L_k = L_0

        history = defaultdict(list) if trace else None
        status = ""

        # Main loop
        for k in range(max_iter + 1):
            if trace:
                history["func"].append(func_k)
                history["grad_sqr_norm"].append(grad_k_norm_sqr)
                history["grad_norm"].append(g_k_norm)
                history["L"].append(L_k)
                history["func_calls"].append(oracle.func_calls)
                history["grad_calls"].append(oracle.grad_calls)
                history["time"].append(
                    (datetime.now() - start_timestamp).total_seconds()
                )

            if k == max_iter:
                status = "iterations_exceeded"
                break

            line_search_max_iter = 30
            for i in range(line_search_max_iter + 1):
                if i == line_search_max_iter:
                    if warnings:
                        print("W: line_search_max_iter_reached", flush=True)
                    break

                T = x_k - precond(grad_k) / L_k
                func_T = oracle.func(T)
                history["func_at_lssolve"].append(func_T)
                grad_new = oracle.grad(T)
                grad_new_norm_sqr = dual_norm_sqr(grad_new)
                history["grad_norm_at_lssolve"].append(grad_new_norm_sqr**0.5)

                if not line_search:
                    break

                if func_T <= func_k + grad_k.dot(T - x_k) + 0.5 * L_k * l2_norm_sqr(
                    T - x_k
                ):
                    L_k *= 0.5
                    L_k = max(L_k, L_min)
                    break

                L_k *= 2

            x_k = T
            func_k = func_T
            grad_k = oracle.grad(T)
            grad_k_norm_sqr = dual_norm_sqr(grad_k)
            g_k_norm = np.sqrt(grad_k_norm_sqr)

        return x_k, status, history

    def leapssn(
        self,
        x_0,
        n_iters=1000,
        Lambda_0=1.0,
        alpha=0.5,
        beta=0.25,
        adaptive_search=True,
        adaptive_search_max_iter=100_000,
        trace=True,
        B=None,
        Binv=None,
        eps=1e-8,
        f_star=None,
        grad_tol=None,
        warnings=False,
        accept_tol=1e-8,
    ):
        oracle = OracleCallsCounter(self)
        start_timestamp = datetime.now()

        # Initialization of the dual norm
        if B is None:
            B = np.eye(x_0.shape[0])
            # dual_norm_sqr = lambda x: np.linalg.norm(x, 2) ** 2
            dual_norm_sqr = lambda x: x.dot(x)
        else:
            if Binv is None and B is None:
                Binv = np.eye(x_0.shape[0])
            elif Binv is None:
                warnings.warning(
                    "Numerically computing an inverse matrix. "
                    "This is numerically unstable and slow. Consider passing in the Riesz map directly."
                )
                Binv = np.linalg.inv(B)
            dual_norm_sqr = lambda x: Binv.dot(x).dot(x)

        # Initialization of the method
        x_k = np.copy(x_0)
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        # print("Residual norm: %s"%(np.linalg.norm(g_k)))
        g_k_norm = np.sqrt(dual_norm_sqr(g_k))
        Lambda_k = Lambda_0

        history = defaultdict(list) if trace else None
        matrix_inverses = 0
        status = ""

        # Main loop
        no_backtracks = 0
        for k in range(n_iters + 1):
            if trace:
                history["func"].append(f_k)
                history["grad_norm"].append(g_k_norm)
                history["Lambda_k"].append(Lambda_k)
                history["grad_calls"].append(oracle.grad_calls)
                history["matrix_inverses"].append(matrix_inverses)
                history["hess_calls"].append(oracle.hess_calls)
                history["time"].append(
                    (datetime.now() - start_timestamp).total_seconds()
                )

            if (f_star is not None and f_k - f_star < eps) or (
                grad_tol is not None and g_k_norm < grad_tol
            ):
                status = "success, %d iters" % k
                break

            if k == n_iters:
                status = "iterations_exceeded"
                break

            Hess_k = oracle.hess(x_k)
            f_current = oracle.func(x_k)

            history["func_at_hess_eval"].append(oracle.func(x_k))

            for i in range(adaptive_search_max_iter + 1):
                if i == adaptive_search_max_iter:
                    if warnings:
                        print(
                            ("W: adaptive_iterations_exceeded, k = %d" % k), flush=True
                        )
                    break

                lambda_k = Lambda_k
                # print(lambda_k)
                try:
                    # Compute the regularized Newton step

                    delta_x = scipy.linalg.cho_solve(
                        scipy.linalg.cho_factor(Hess_k + lambda_k * B, lower=False),
                        -g_k,
                    )
                    # delta_x = np.linalg.solve(Hess_k + lambda_k * B, -g_k)
                    matrix_inverses += 1

                except (np.linalg.LinAlgError, ValueError) as e:
                    if warnings:
                        print("W: linalg_error", flush=True)
                f_new = oracle.func(x_k + delta_x)
                history["func_at_lssolve"].append(f_new)

                g_new = oracle.grad(x_k + delta_x)
                g_new_norm_sqr = dual_norm_sqr(g_new)
                history["grad_norm_at_lssolve"].append(g_new_norm_sqr**0.5)

                if not adaptive_search:
                    break

                # Check condition for Lambda_k
                c1 = g_new.dot(-delta_x)
                c2 = alpha * g_new_norm_sqr / lambda_k
                c3 = f_current - f_new
                c4 = beta * lambda_k * delta_x.dot((B.dot(delta_x)))
                # print("c1=%s, c2=%s, c3=%s, c4=%s"%(c1,c2,c3,c4))

                if ((c1 >= c2) and (c3 >= c4)) or (
                    np.abs(c1) <= accept_tol
                    and np.abs(c2) <= accept_tol
                    and np.abs(c3) <= accept_tol
                    and np.abs(c4) <= accept_tol
                ):
                    history["func_at_acceptance"].append(f_new)
                    history["grad_norm_at_acceptance"].append(g_new_norm_sqr**0.5)

                    # print("Update accepted.")
                    Lambda_k = Lambda_k * 0.5
                    break
                Lambda_k = Lambda_k * 2
                no_backtracks += 1

            history["reg_parameter"].append(lambda_k)

            # Update the point
            x_k += delta_x
            f_k = f_new
            g_k = g_new
            # print("Residual norm: %s"%(np.linalg.norm(g_k)))
            g_k_norm = np.sqrt(g_new_norm_sqr)
        print("No of backtracks: ", no_backtracks)
        return x_k, status, history

    def gladssn(
        self,
        x_0,
        n_iters=1000,
        Lambda_0=1.0,
        alpha=0.5,
        beta=0.25,
        adaptive_search=True,
        adaptive_search_max_iter=100_000,
        trace=True,
        B=None,
        Binv=None,
        eps=1e-8,
        f_star=None,
        grad_tol=None,
        warnings=False,
        accept_tol=1e-8,
        lazy_Hessians=False,
        update_freq=1,
    ):
        oracle = OracleCallsCounter(self)
        start_timestamp = datetime.now()

        # Initialization of the dual norm
        if B is None:
            B = np.eye(x_0.shape[0])
            # dual_norm_sqr = lambda x: np.linalg.norm(x, 2) ** 2
            dual_norm_sqr = lambda x: x.dot(x)
        else:
            if Binv is None and B is None:
                Binv = np.eye(x_0.shape[0])
            elif Binv is None:
                warnings.warning(
                    "Numerically computing an inverse matrix. "
                    "This is numerically unstable and slow. Consider passing in the Riesz map directly."
                )
                Binv = np.linalg.inv(B)
            dual_norm_sqr = lambda x: Binv.dot(x).dot(x)

        # Initialization of the method
        x_k = np.copy(x_0)
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        # print("Residual norm: %s"%(np.linalg.norm(g_k)))
        g_k_norm = np.sqrt(dual_norm_sqr(g_k))
        Lambda_k = Lambda_0

        history = defaultdict(list) if trace else None
        matrix_inverses = 0
        status = ""

        # Main loop
        no_backtracks = 0
        for k in range(n_iters + 1):
            if trace:
                history["func"].append(f_k)
                history["grad_norm"].append(g_k_norm)
                history["Lambda_k"].append(Lambda_k)
                history["grad_calls"].append(oracle.grad_calls)
                history["matrix_inverses"].append(matrix_inverses)
                history["hess_calls"].append(oracle.hess_calls)
                history["time"].append(
                    (datetime.now() - start_timestamp).total_seconds()
                )

            if (f_star is not None and f_k - f_star < eps) or (
                grad_tol is not None and g_k_norm < grad_tol
            ):
                status = "success, %d iters" % k
                break

            if k == n_iters:
                status = "iterations_exceeded"
                break

            if lazy_Hessians is True and k % update_freq != 0:
                pass
                # print("Didn't update Hessian at iteration " + str(k))
            else:
                Hess_k = oracle.hess(x_k)

            # Hess_k = oracle.hess(x_k)
            f_current = oracle.func(x_k)

            history["func_at_hess_eval"].append(oracle.func(x_k))

            for i in range(adaptive_search_max_iter + 1):
                if i == adaptive_search_max_iter:
                    if warnings:
                        print(
                            ("W: adaptive_iterations_exceeded, k = %d" % k), flush=True
                        )
                    break

                lambda_k = Lambda_k * g_k_norm**0.5
                # print(lambda_k)
                try:
                    # Compute the regularized Newton step

                    delta_x = scipy.linalg.cho_solve(
                        scipy.linalg.cho_factor(Hess_k + lambda_k * B, lower=False),
                        -g_k,
                    )
                    # delta_x = np.linalg.solve(Hess_k + lambda_k * B, -g_k)
                    matrix_inverses += 1

                except (np.linalg.LinAlgError, ValueError) as e:
                    if warnings:
                        print("W: linalg_error", flush=True)
                f_new = oracle.func(x_k + delta_x)
                history["func_at_lssolve"].append(f_new)

                g_new = oracle.grad(x_k + delta_x)
                g_new_norm_sqr = dual_norm_sqr(g_new)
                history["grad_norm_at_lssolve"].append(g_new_norm_sqr**0.5)

                if not adaptive_search:
                    break

                # Check condition for Lambda_k
                c1 = g_new.dot(-delta_x)
                c2 = alpha * g_new_norm_sqr / lambda_k
                c3 = f_current - f_new
                c4 = beta * lambda_k * delta_x.dot((B.dot(delta_x)))
                # print("c1=%s, c2=%s, c3=%s, c4=%s"%(c1,c2,c3,c4))

                if ((c1 >= c2) and (c3 >= c4)) or (
                    np.abs(c1) <= accept_tol
                    and np.abs(c2) <= accept_tol
                    and np.abs(c3) <= accept_tol
                    and np.abs(c4) <= accept_tol
                ):
                    history["func_at_acceptance"].append(f_new)
                    history["grad_norm_at_acceptance"].append(g_new_norm_sqr**0.5)

                    # print("Update accepted.")
                    Lambda_k = Lambda_k * 0.25
                    break
                Lambda_k = Lambda_k * 4
                no_backtracks += 1

            history["reg_parameter"].append(lambda_k)

            # Update the point
            x_k += delta_x
            f_k = f_new
            g_k = g_new
            # print("Residual norm: %s"%(np.linalg.norm(g_k)))
            g_k_norm = np.sqrt(g_new_norm_sqr)
        print("No of backtracks: ", no_backtracks)
        return x_k, status, history


    def svrg(
    oracle,
    x_0,
    n_epochs=20,
    m=None,
    eta=1e-10,
    trace=True,
):
        oracle = OracleCallsCounter(oracle)
        base_oracle = oracle.oracle
        start_timestamp = datetime.now()

        x_k = np.copy(x_0)
        n = base_oracle.x.shape[0]

        if m is None:
            m = n

        history = defaultdict(list) if trace else None

        for k in range(n_epochs):
            g_k = oracle.grad(x_k)

            if trace:
                f_val = oracle.func(x_k)
                history["func"].append(f_val)
                history["grad_norm"].append(np.linalg.norm(g_k))
                history["time"].append(
                    (datetime.now() - start_timestamp).total_seconds()
                )
                history["grad_calls"].append(oracle.grad_calls)
                history["func_calls"].append(oracle.func_calls)

            x_t = x_k.copy()

            for t in range(m):
                i = np.random.randint(n)

                grad_it = base_oracle.grad_i(x_t, i)
                grad_i_snapshot = base_oracle.grad_i(x_k, i)

                v_t = grad_it - grad_i_snapshot + g_k
                x_t = x_t - eta * v_t

                if not np.all(np.isfinite(x_t)):
                    return x_t, "diverged", history

            x_k = x_t

        return x_k, "success", history
    
    def svrg_bb(
    oracle,
    x_0,
    n_epochs=50,
    m=None,
    eta_0=1e-10,
    trace=True,
): # SVRG-BB algorithm from https://arxiv.org/pdf/1605.04131
        oracle = OracleCallsCounter(oracle)
        base_oracle = oracle.oracle
        start_timestamp = datetime.now()

        x_k = np.copy(x_0)
        n = base_oracle.x.shape[0]

        if m is None:
            m = n

        history = defaultdict(list) if trace else None
        status = ""

        x_prev = None
        g_prev = None
        eta_k = eta_0

        for k in range(n_epochs + 1):
            # Snapshot full gradient
            g_k = oracle.grad(x_k)
            g_k_norm = np.linalg.norm(g_k)

            if trace:
                history["func"].append(oracle.func(x_k))
                history["grad_norm"].append(g_k_norm)
                history["eta"].append(eta_k)
                history["time"].append(
                    (datetime.now() - start_timestamp).total_seconds()
                )
                history["grad_calls"].append(oracle.grad_calls)
                history["func_calls"].append(oracle.func_calls)

            if not np.all(np.isfinite(x_k)) or not np.all(np.isfinite(g_k)):
                status = "diverged"
                break

            if k == n_epochs:
                status = "iterations_exceeded"
                break

            # BB stepsize update
            if k == 0:
                eta_k = eta_0
            else:
                s = x_k - x_prev
                y_vec = g_k - g_prev
                denom = float(s.dot(y_vec))

                eta_bb = s.dot(s) / (m * denom)
                eta_k = eta_bb

            x_prev = x_k.copy()
            g_prev = g_k.copy()

            # Inner loop
            x_t = x_k.copy()
            for t in range(m):
                i = np.random.randint(n)

                grad_it = base_oracle.grad_i(x_t, i)
                grad_i_snap = base_oracle.grad_i(x_k, i)

                v_t = grad_it - grad_i_snap + g_k

                if not np.all(np.isfinite(v_t)):
                    status = "diverged"
                    break

                x_t = x_t - eta_k * v_t
                # history["func_at_lssolve"].append(oracle.func(x_t))

                if not np.all(np.isfinite(x_t)):
                    status = "diverged"
                    break

            if status == "diverged":
                if warnings:
                    print(f"W: svrg_bb diverged at epoch {k}", flush=True)
                break

            x_k = x_t

        return x_k, status, history