"""Checks for the opt-in exact input-Hessian penalty (CPU float64)."""
import unittest

import torch
from torch import nn

from NN.exact_spectral_norm import (
    ActiveSpectralTieError, exact_hessian_norm_penalty, input_hessians_2d,
    smoothed_squared_spectral_norm_2d,
)


class MatrixQuadratic(nn.Module):
    def forward(self, x):
        return (0.5 * ((x @ self.matrix) * x).sum(dim=1)).unsqueeze(1)

    def __init__(self, matrix):
        super().__init__()
        self.matrix = nn.Parameter(torch.tensor(matrix, dtype=torch.float64))


class TanhRidge(nn.Module):
    def __init__(self):
        super().__init__()
        self.vector = nn.Parameter(torch.tensor([1.2, -0.7, 0.3, 3.0], dtype=torch.float64))

    def forward(self, x):
        w, b, a = self.vector[:2], self.vector[2], self.vector[3]
        return (a * torch.tanh(x @ w + b)).unsqueeze(1)


class ExactSpectralChecks(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[0.1, 0.2], [0.3, -0.1], [-0.2, 0.1]], dtype=torch.float64)

    def test_negative_dominant_eigenvalue_and_coefficient(self):
        model = MatrixQuadratic([[-2.0, 0.0], [0.0, 0.5]])
        h = input_hessians_2d(model, dict(model.named_parameters()), self.x)
        torch.testing.assert_close(h, model.matrix.expand(3, 2, 2))
        p, maximum = exact_hessian_norm_penalty(model, dict(model.named_parameters()), self.x, coeff=2.0)
        self.assertEqual(maximum.item(), 2.0)
        self.assertEqual(p.item(), 9.0)

    def test_batch_mean_and_sample_max(self):
        model = TanhRidge()
        def evaluate(x):
            return exact_hessian_norm_penalty(model, dict(model.named_parameters()), x, coeff=1.3)
        p, maximum = evaluate(self.x)
        parts = [evaluate(x[None]) for x in self.x]
        torch.testing.assert_close(p, torch.stack([v[0] for v in parts]).mean())
        torch.testing.assert_close(maximum, torch.stack([v[1] for v in parts]).max())
        duplicate, duplicate_max = evaluate(self.x.repeat(2, 1))
        torch.testing.assert_close(p, duplicate)
        torch.testing.assert_close(maximum, duplicate_max)

    def test_nonlinear_input_hessian_and_parameter_derivatives(self):
        model = TanhRidge()
        vector = model.vector.detach().requires_grad_(True)
        def objective(v):
            return exact_hessian_norm_penalty(model, {'vector': v}, self.x, coeff=1.3)[0]
        def reference(v):
            w, b, a = v[:2], v[2], v[3]
            t = torch.tanh(self.x @ w + b)
            scale = -2 * a * t * (1-t.square())
            q = scale.square() * w.square().sum().square()
            return 0.5 * 1.3 * torch.relu(q-1).square().mean()
        t = torch.tanh(self.x @ vector[:2] + vector[2])
        scale = -2 * vector[3] * t * (1-t.square())
        href = scale[:, None, None] * torch.outer(vector[:2], vector[:2])[None]
        torch.testing.assert_close(input_hessians_2d(model, {'vector':vector}, self.x), href)
        self.assertGreater(objective(vector).item(), 0.0)  # Exercise active penalties.
        torch.testing.assert_close(objective(vector), reference(vector))
        torch.testing.assert_close(torch.autograd.functional.jacobian(objective, vector),
                                   torch.autograd.functional.jacobian(reference, vector), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(torch.autograd.functional.hessian(objective, vector),
                                   torch.autograd.functional.hessian(reference, vector), rtol=1e-9, atol=1e-9)
        self.assertTrue(torch.autograd.gradcheck(objective, (vector,), eps=1e-6, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.autograd.gradgradcheck(objective, (vector,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_inactive_repeated_eigenvalues_have_zero_derivatives(self):
        for value in [0.0, 0.5, 1.0]:
            model = MatrixQuadratic([[value, 0.0], [0.0, value]])
            def objective(matrix):
                return exact_hessian_norm_penalty(model, {'matrix':matrix}, self.x)[0]
            self.assertEqual(objective(model.matrix).item(), 0.0)
            g = torch.autograd.functional.jacobian(objective, model.matrix)
            h = torch.autograd.functional.hessian(objective, model.matrix)
            self.assertTrue(torch.equal(g, torch.zeros_like(g)))
            self.assertTrue(torch.equal(h, torch.zeros_like(h)))

    def test_affine_network_has_zero_input_hessian(self):
        model = nn.Linear(2, 1, dtype=torch.float64)
        p, maximum = exact_hessian_norm_penalty(model, dict(model.named_parameters()), self.x)
        self.assertEqual(p.item(), 0.0)
        self.assertEqual(maximum.item(), 0.0)

    def test_active_equal_magnitude_eigenvalues_are_reported(self):
        for second in [2.0, -2.0]:
            model = MatrixQuadratic([[2.0, 0.0], [0.0, second]])
            with self.assertRaisesRegex(ActiveSpectralTieError, 'sample indices'):
                exact_hessian_norm_penalty(model, dict(model.named_parameters()), self.x, tie_policy='raise')

    def test_default_ties_keep_values_and_finite_averaged_derivatives(self):
        for second in [2.0, -2.0, 2.0+1e-15]:
            model = MatrixQuadratic([[2.0, 0.0], [0.0, second]])
            def objective(matrix):
                return exact_hessian_norm_penalty(model, {'matrix':matrix}, self.x)[0]
            p, maximum = exact_hessian_norm_penalty(model, dict(model.named_parameters()), self.x)
            exact_q = torch.linalg.eigvalsh(model.matrix.detach()).square().max()
            torch.testing.assert_close(p.detach(), .5*(exact_q-1).square(), rtol=0, atol=1e-14)
            torch.testing.assert_close(maximum, exact_q.sqrt(), rtol=0, atol=0)
            g = torch.autograd.functional.jacobian(objective, model.matrix)
            h = torch.autograd.functional.hessian(objective, model.matrix)
            # Mean of the two squared-eigenvalue gradients: grad(q)=B.
            torch.testing.assert_close(g, (exact_q-1)*model.matrix.detach())
            self.assertTrue(torch.isfinite(h).all())
            # An averaged direction lowers both equally active eigenvalue norms.
            self.assertLess(objective(model.matrix.detach()-1e-3*g).item(), p.item())

    def test_continue_and_strict_agree_away_from_ties(self):
        model = TanhRidge()
        def evaluate(v, policy):
            return exact_hessian_norm_penalty(model, {'vector':v}, self.x, tie_policy=policy)[0]
        self.assertEqual(evaluate(model.vector, 'average').item(), evaluate(model.vector, 'raise').item())
        for derivative in [torch.autograd.functional.jacobian, torch.autograd.functional.hessian]:
            torch.testing.assert_close(derivative(lambda v:evaluate(v,'average'), model.vector),
                                       derivative(lambda v:evaluate(v,'raise'), model.vector), rtol=0, atol=0)

    def test_optimizer_can_step_from_active_spectral_tie(self):
        from NN.GLAdSSNOptimizer_new import GlAdSSNOptimizer
        from NN.utils_new import make_param_vector_interface
        model = MatrixQuadratic([[-2.0, 0.0], [0.0, 2.0]])
        pack, unpack, assign, _ = make_param_vector_interface(model)
        def objective(v):
            return exact_hessian_norm_penalty(model, unpack(v), self.x)[0]
        opt = GlAdSSNOptimizer(model.parameters(), m=25, number_of_steps_to_freeze=2, solve_rtol=1e-10)
        previous = objective(pack()).item()
        for _ in range(3):
            f, g, _ = opt.step(objective, pack, assign)
            self.assertEqual(opt.state['status'], 'accepted')
            self.assertTrue(torch.isfinite(g).all())
            self.assertLess(f.item(), previous)
            self.assertLessEqual(opt.state['last_linear_residual'], 1e-10)
            previous = f.item()

    def test_spectral_smoothing_error_bound(self):
        generator = torch.Generator().manual_seed(871)
        raw = torch.randn(30,2,2,dtype=torch.float64,generator=generator)
        matrices = .5*(raw+raw.transpose(-1,-2))
        exact = torch.linalg.eigvalsh(matrices).square().amax(-1)
        for epsilon in [1e-4,1e-3,1e-2]:
            q = smoothed_squared_spectral_norm_2d(matrices,epsilon)
            self.assertTrue(torch.all(q>=exact-1e-12))
            self.assertTrue(torch.all(q<=exact+epsilon/4+1e-12))
            gap=(torch.linalg.eigvalsh(matrices).square()[:,1]-torch.linalg.eigvalsh(matrices).square()[:,0]).abs()
            torch.testing.assert_close(q[gap>=epsilon],exact[gap>=epsilon],rtol=1e-12,atol=1e-12)
        repeated = 2*torch.eye(2,dtype=torch.float64)
        self.assertAlmostEqual(smoothed_squared_spectral_norm_2d(repeated,.01).item(),4.0025)

    def test_smoothed_penalty_derivatives_at_spectral_ties(self):
        model = MatrixQuadratic([[2.,0.],[0.,2.]])
        for second in [2., -2., 2.+1e-15]:
            v=torch.tensor([2.,0.,second],dtype=torch.float64,requires_grad=True)
            def objective(vector):
                matrix=torch.stack([torch.stack([vector[0],vector[1]]),torch.stack([vector[1],vector[2]])])
                return exact_hessian_norm_penalty(
                    model, {'matrix':matrix}, self.x, spectral_smoothing_eps=.01,
                    tie_policy='raise')[0]  # Smoothing removes the tie exception.
            self.assertTrue(torch.autograd.gradcheck(objective,(v,),eps=1e-6,atol=1e-5,rtol=1e-4))
            self.assertTrue(torch.autograd.gradgradcheck(objective,(v,),eps=1e-6,atol=1e-3,rtol=1e-4))
        p,maximum=exact_hessian_norm_penalty(model,dict(model.named_parameters()),self.x,spectral_smoothing_eps=.01)
        self.assertEqual(maximum.item(),2.)
        self.assertAlmostEqual(p.item(),.5*(4.0025-1)**2)

    def test_smoothed_zero_hessian_derivatives_are_finite(self):
        model=MatrixQuadratic([[0.,0.],[0.,0.]])
        def objective(m):
            return exact_hessian_norm_penalty(model,{'matrix':m},self.x,spectral_smoothing_eps=.001)[0]
        self.assertEqual(objective(model.matrix).item(),0.)
        for derivative in [torch.autograd.functional.jacobian,torch.autograd.functional.hessian]:
            value=derivative(objective,model.matrix)
            self.assertTrue(torch.equal(value,torch.zeros_like(value)))

    def test_huber_band_boundary_has_continuous_gradient_and_hessian_jump(self):
        epsilon=.1
        # Eigenvalues of B^2 are 2+t and 2, so its gap is t near epsilon.
        def objective(t):
            matrix=torch.diag(torch.stack([(2+t).sqrt(),torch.sqrt(t*0+2)]))
            return smoothed_squared_spectral_norm_2d(matrix,epsilon)
        gradients=[];hessians=[]
        for delta in [-1e-7,1e-7]:
            t=torch.tensor(epsilon+delta,dtype=torch.float64,requires_grad=True)
            gradients.append(torch.autograd.functional.jacobian(objective,t))
            hessians.append(torch.autograd.functional.hessian(objective,t))
        self.assertLess(abs(gradients[0].item()-gradients[1].item()),1e-6)
        self.assertAlmostEqual(gradients[1].item(),1.,places=10)
        self.assertAlmostEqual(hessians[0].item(),1/(2*epsilon),places=8)
        self.assertAlmostEqual(hessians[1].item(),0.,places=8)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
