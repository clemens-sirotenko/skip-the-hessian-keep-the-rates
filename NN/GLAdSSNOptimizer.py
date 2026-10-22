import torch
# -------------------------
# Utilities
# -------------------------


def make_param_spec(model, trainable_only: bool = True):
    """
    Creates a consistent mapping between:
      - the model's (trainable) parameters
      - a single flattened parameter vector

    Returns:
      pack() -> (P,) tensor
      vec_to_param_dict(vec) -> {param_name: tensor_view}
      assign(vec) -> in-place copy vec -> model parameters
      total -> number of scalars in vec
    """
    names, shapes, numels, params = [], [], [], []

    for name, p in model.named_parameters():
        if (not trainable_only) or p.requires_grad:
            names.append(name)
            shapes.append(p.shape)
            numels.append(p.numel())
            params.append(p)

    total = sum(numels)

    def pack():
        # Current model params -> one flat vector (detached)
        return torch.cat([p.detach().reshape(-1) for p in params])

    def vec_to_param_list(vec):
        # Flat vector -> list of tensors (views into vec)
        out, off = [], 0
        for n, shape in zip(numels, shapes):
            out.append(vec[off : off + n].view(shape))
            off += n
        if off != vec.numel():
            raise ValueError(f"Vector has {vec.numel()} elements, expected {off}.")
        return out

    def vec_to_param_dict(vec):
        # Flat vector -> {name: tensor_view}
        return {k: v for k, v in zip(names, vec_to_param_list(vec))}

    @torch.no_grad()
    def assign(vec):
        # In-place update of model parameters from vec
        off = 0
        for p, n, shape in zip(params, numels, shapes):
            p.copy_(vec[off : off + n].view(shape))
            off += n
        if off != vec.numel():
            raise ValueError(f"Vector has {vec.numel()} elements, expected {off}.")

    return pack, vec_to_param_dict, assign, total


class GlAdSSNOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        alpha=0.5,
        beta=0.25,
        m=4,
        lambda0=1.0,
        number_of_steps_to_freeze=1,
        lam_min=1e-8,
        symmetrize=True,
    ):
        defaults = dict(
            alpha=alpha,
            beta=beta,
            m=m,
            number_of_steps_to_freeze=number_of_steps_to_freeze,
            lam_min=lam_min,
            symmetrize=symmetrize,
        )
        super().__init__(params, defaults)
        self.state["lambda_k"] = float(lambda0)
        self.state["k"] = 0
        self.state["freeze_counter"] = 0
        self.state["H"] = None

    def step(self, loss_from_vec, pack, assign):
        group = self.param_groups[0]
        alpha = group["alpha"]
        beta = group["beta"]
        m = group["m"]
        number_of_steps_to_freeze = group["number_of_steps_to_freeze"]
        symmetrize = group["symmetrize"]

        xk = pack().detach()

        # Evaluate f,g at current x
        x_var = xk.detach().requires_grad_(True)
        fk = loss_from_vec(x_var)
        gk = torch.autograd.grad(fk, x_var, create_graph=False)[0].detach()
        gk_norm_squared = float(gk.dot(gk))

        # (Re)compute Hessian
        if (
            self.state["H"] is None
            or self.state["freeze_counter"] >= number_of_steps_to_freeze
        ):
            H = torch.autograd.functional.hessian(loss_from_vec, x_var).detach()
            if symmetrize:
                H = 0.5 * (H + H.T)
            self.state["H"] = H
            self.state["freeze_counter"] = 0

        H = self.state["H"]
        self.state["freeze_counter"] += 1

        lambda_k = float(self.state["lambda_k"])
        n = xk.numel()
        Id = torch.eye(n, device=xk.device, dtype=xk.dtype)

        accepted = False
        x_plus = xk
        chosen_lam = lambda_k

        for j in range(m + 1):
            # #lam = max(lam_min, (4.0 ** j) * lambda_k * (gnorm_sq ** 0.25 if gnorm_sq > 0 else 1.0))
            # lam = max(lam_min, (4.0 ** j) * lambda_k )#* gnorm_sq ** 0.25)
            lam = (4.0**j) * lambda_k * gk_norm_squared**0.25
            # lam = max(lam_min, (4.0 ** j) * lambda_k * gnorm_sq ** 0.25)
            A = H + lam * Id

            # Solve (H + lam Id) s = g
            try:
                s = torch.linalg.solve(A, gk)
            except RuntimeError:
                # fallback if A is singular / ill-conditioned
                s = torch.linalg.lstsq(A, gk.unsqueeze(1)).solution.squeeze(1)

            x_cand = xk - s

            # Candidate loss and gradient
            x_cand_var = x_cand.detach().requires_grad_(True)
            f_cand = loss_from_vec(x_cand_var)
            g_cand = torch.autograd.grad(f_cand, x_cand_var, create_graph=False)[
                0
            ].detach()

            step = xk - x_cand  # = s
            Fprime_term = g_cand.dot(step)
            g_cand_norm_squared = g_cand.dot(g_cand)

            # Acceptance criteria
            cond1 = Fprime_term >= (alpha / lam) * g_cand_norm_squared
            cond2 = (fk.detach() - f_cand.detach()) >= beta * lam * step.dot(step)

            if cond1 and cond2:
                accepted = True
                x_plus = x_cand.detach()
                chosen_lam = lam
                break

        if accepted:
            assign(x_plus)
            self.state["lambda_k"] = chosen_lam / 4.0
        else:
            self.state["lambda_k"] = lambda_k * (4.0 ** (m + 1))

        self.state["k"] += 1
        return fk.detach(), gk.detach(), self.state["lambda_k"]
