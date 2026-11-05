"""Focused numerical checks for the opt-in corrected NN implementation.

From the repository root:
    python -m unittest discover -s NN -p test_nn_corrections.py
"""
import unittest
from unittest.mock import patch

import torch

from NN.GLAdSSNOptimizer_new import GlAdSSNOptimizer
from NN.utils_new import hessian_norm_penalty


class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(1.2, dtype=torch.float64))

    def forward(self, x):
        return (0.5 * (self.a * x[:, 0]**2 + 1.1 * x[:, 1]**2)).unsqueeze(1)


class VaryingHessian(torch.nn.Module):
    def forward(self, x):
        # Input Hessian diag(x[0], .2), with known sample spectral norms.
        return (x[:, 0]**3 / 6 + 0.1 * x[:, 1]**2).unsqueeze(1)


class PenaltyChecks(unittest.TestCase):
    def test_per_sample_penalty_and_negative_eigenvalue(self):
        x = torch.tensor([[-2., 0.], [.5, 0.]], dtype=torch.float64)
        penalty, maximum = hessian_norm_penalty(
            VaryingHessian(), {}, x, torch.ones_like(x), coeff=1., K=30)
        # Norms are 2 and .5: (1/(2*2)) * ((4-1)^2 + 0) = 2.25.
        self.assertAlmostEqual(penalty.item(), 2.25, places=11)
        self.assertAlmostEqual(maximum.item(), 2., places=11)

    def test_batch_splitting_and_duplication_preserve_mean(self):
        x = torch.tensor([[-2., .1], [1.4, .3], [.5, .2]], dtype=torch.float64)
        def evaluate(batch):
            return hessian_norm_penalty(
                VaryingHessian(), {}, batch, torch.ones_like(batch), coeff=1., K=5)
        full, maximum = evaluate(x)
        individual = [evaluate(row[None]) for row in x]
        torch.testing.assert_close(full, torch.stack([v[0] for v in individual]).mean())
        torch.testing.assert_close(maximum, torch.stack([v[1] for v in individual]).max())
        duplicate, duplicate_max = evaluate(x.repeat(2, 1))
        torch.testing.assert_close(full, duplicate)
        torch.testing.assert_close(maximum, duplicate_max)

    def test_first_and_second_derivatives_of_finite_surrogate(self):
        model = Quadratic()
        x = torch.tensor([[.2, .3], [.4, -.1]], dtype=torch.float64)
        def objective(a):
            return hessian_norm_penalty(
                model, {'a': a}, x, torch.ones_like(x), coeff=1., K=5)[0]
        a = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(objective, (a,), eps=1e-6, atol=1e-6))
        self.assertTrue(torch.autograd.gradgradcheck(objective, (a,), eps=1e-6, atol=1e-5))

    def test_zero_hessian_has_finite_penalty_derivatives(self):
        class ScaledQuadratic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
            def forward(self, x):
                return self.a * x.square().sum(dim=1, keepdim=True) / 2
        model = ScaledQuadratic()
        x = torch.ones((2, 2), dtype=torch.float64)
        def objective(a):
            return hessian_norm_penalty(
                model, {'a': a}, x, x, coeff=1., K=5)[0]
        a = model.a
        self.assertEqual(objective(a).item(), 0.)
        self.assertEqual(torch.autograd.functional.jacobian(objective, a).item(), 0.)
        self.assertEqual(torch.autograd.functional.hessian(objective, a).item(), 0.)

    def test_configurable_normalization_epsilon(self):
        model = Quadratic()
        x = torch.tensor([[.2, .3], [.4, -.1]], dtype=torch.float64)
        def objective(a, epsilon):
            return hessian_norm_penalty(
                model, {'a': a}, x, torch.ones_like(x), coeff=1., K=5,
                normalization_eps=epsilon)[0]
        a = model.a.detach().requires_grad_(True)
        default, _ = hessian_norm_penalty(model, {'a': a}, x, torch.ones_like(x), coeff=1., K=5)
        torch.testing.assert_close(default, objective(a,1e-12), rtol=0, atol=0)
        self.assertLess(objective(a,.1).item(), default.item())
        self.assertTrue(torch.autograd.gradcheck(lambda v:objective(v,.01), (a,)))
        self.assertTrue(torch.autograd.gradgradcheck(lambda v:objective(v,.01), (a,)))
        for epsilon in [0., -1., float('nan')]:
            with self.assertRaises(ValueError):objective(a,epsilon)


class OptimizerChecks(unittest.TestCase):
    def problem(self, value, **options):
        p = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
        optimizer = GlAdSSNOptimizer([p], **options)
        def assign(x):
            with torch.no_grad():
                p.copy_(x)
        return p, optimizer, lambda: p.detach().clone(), assign

    def test_coefficient_and_returned_metrics(self):
        p, opt, pack, assign = self.problem([4.])
        f, g, coefficient = opt.step(lambda x: .5*x.square().sum(), pack, assign)
        self.assertEqual(coefficient, .25)
        self.assertEqual(opt.state['accepted_lam'], 2.)
        torch.testing.assert_close(g, p.detach())
        torch.testing.assert_close(f, .5*p.detach().square().sum())
        self.assertEqual(opt.state['status'], 'accepted')

    def test_zero_gradient_stops_before_hessian(self):
        p, opt, pack, assign = self.problem([0.])
        opt.step(lambda x: .5*x.square().sum(), pack, assign)
        self.assertEqual(opt.state['status'], 'converged')
        self.assertEqual(opt.state['hessian_evaluations'], 0)
        self.assertEqual(opt.state['k'], 0)
        self.assertEqual(p.item(), 0.)

    def test_failed_search_does_not_advance_iterate_or_lazy_schedule(self):
        p, opt, pack, assign = self.problem([.1], m=0, number_of_steps_to_freeze=5)
        opt.step(lambda x: (x.square()-1).square().sum(), pack, assign)
        self.assertEqual(opt.state['status'], 'line_search_failed')
        self.assertEqual(p.item(), .1)
        self.assertEqual(opt.state['k'], 0)
        self.assertEqual(opt.state['freeze_counter'], 0)
        self.assertEqual(opt.state['lambda_k'], 1.)

    def test_hessian_refresh_counts_accepted_updates(self):
        _, opt, pack, assign = self.problem([4.], number_of_steps_to_freeze=2)
        for expected in [1, 1, 2]:
            opt.step(lambda x: .5*x.square().sum(), pack, assign)
            self.assertEqual(opt.state['status'], 'accepted')
            self.assertEqual(opt.state['hessian_evaluations'], expected)
        self.assertEqual(opt.state['k'], 3)

    def test_inconsistent_singular_solve_is_retried(self):
        _, opt, pack, assign = self.problem([0.])
        # First trial: A = H+lambda = -1+1 = 0, but g=1.
        opt.step(lambda x: (-.5*x.square()+x).sum(), pack, assign)
        self.assertEqual(opt.state['status'], 'accepted')
        self.assertEqual(opt.state['last_trials'], 2)
        self.assertEqual(opt.state['linear_solves'], 3)
        self.assertLess(opt.state['last_linear_residual'], 1e-12)

    def test_consistent_singular_least_squares_can_be_accepted(self):
        _, opt, pack, assign = self.problem([0., 0.])
        # The default QR driver can report either rank 0 or 1 for this singular
        # fixture on the Windows backend. Use SVD only in this test to exercise
        # acceptance of an accurate least-squares fallback deterministically.
        original_lstsq = torch.linalg.lstsq
        with patch('torch.linalg.lstsq', side_effect=lambda a,b: original_lstsq(a,b,driver='gelsd')):
            opt.step(lambda x: -.5*x[0]**2+.5*x[1]**2+x[1], pack, assign)
        self.assertEqual(opt.state['status'], 'accepted')
        self.assertEqual(opt.state['last_trials'], 1)
        self.assertEqual(opt.state['linear_solves'], 2)
        self.assertLess(opt.state['last_linear_residual'], 1e-12)

    def test_damping_floor_adjusts_effective_coefficient(self):
        _, opt, pack, assign = self.problem([4.], lambda0=1e-4, lam_min=1.)
        _, _, coefficient = opt.step(lambda x: .5*x.square().sum(), pack, assign)
        self.assertEqual(opt.state['accepted_lam'], 1.)
        self.assertEqual(coefficient, .125)

    def test_nonfinite_gradient_is_reported(self):
        p, opt, pack, assign = self.problem([1.])
        opt.step(lambda x: (x * float('nan')).sum(), pack, assign)
        self.assertEqual(opt.state['status'], 'nonfinite_objective_or_gradient')
        self.assertEqual(opt.state['k'], 0)
        self.assertEqual(p.item(), 1.)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
