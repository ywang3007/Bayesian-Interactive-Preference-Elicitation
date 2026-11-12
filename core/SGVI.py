import torch


class SGVI:
    """Stochastic Gradient Variational Inference for Dirichlet-parameterized
    piecewise-linear utility models.

    The posterior over utility weights ``u`` is approximated by a Dirichlet
    distribution with concentration parameter ``phi**2``. ``phi`` is fitted
    by stochastic gradient ascent on the ELBO using the reparameterization
    trick (standard-normal epsilon + softmax).
    """

    def __init__(self, evalmatrix, intervals_num, Q, e, T, W_g, device, temperature=10):
        """
        :param evalmatrix: ``(n, m)`` performance matrix for ``n`` alternatives on ``m`` criteria.
        :param intervals_num: number of sub-intervals per criterion for the piecewise-linear value function.
        :param Q: reference set, a tensor of pairwise comparisons provided by the DM; each row ``(i, j)`` means ``a_i`` preferred to ``a_j``.
        :param e: convergence tolerance on the norm of the gradient / parameter change.
        :param T: maximum number of gradient-ascent iterations.
        :param W_g: Monte-Carlo sample size used to estimate the stochastic gradient.
        :param device: torch device.
        :param temperature: Bradley-Terry inverse temperature.
        """
        self.device = device
        self.evalmatrix = evalmatrix
        self.n = evalmatrix.shape[0]  # number of alternatives
        self.m = evalmatrix.shape[1]  # number of criteria
        self.intervals_num = intervals_num
        self.K = self.m * self.intervals_num  # number of delta_u parameters
        self.Q = Q
        self.e = e
        self.T = T
        self.W_g = W_g
        self.temperature = temperature
        self.G = self.criteria_div()
        self.V = self.v_compute()
        self.alpha = torch.ones(self.K).to(self.device)  # Dirichlet prior concentration

    def criteria_div(self):
        """Return the ``(intervals_num + 1, m)`` grid of sub-interval endpoints."""
        criteria = torch.sort(self.evalmatrix.clone(), dim=0)[0]
        step = (criteria[-1, :] - criteria[0, :]) / self.intervals_num
        G = torch.zeros((self.intervals_num + 1, self.m)).to(self.device)
        for i in range(self.intervals_num + 1):
            G[i, :] = criteria[0, :] + i * step
        return G

    def v_compute(self):
        """Return the ``(K, n)`` piecewise-linear coefficient matrix mapping
        ``delta_u`` to alternative values."""
        V = torch.zeros((self.K, self.n)).to(self.device)
        for i in range(self.n):
            ind = 0
            for j in range(self.m):
                for k in range(self.intervals_num):
                    if self.evalmatrix[i][j] > self.G[k][j] and self.evalmatrix[i][j] >= self.G[k + 1][j]:
                        V[ind, i] = 1
                    elif self.G[k][j] <= self.evalmatrix[i][j] <= self.G[k + 1][j]:
                        V[ind, i] = (self.evalmatrix[i][j] - self.G[k][j]) / (self.G[k + 1][j] - self.G[k][j])
                    ind += 1
        return V

    def loglikelihood(self, u, Q):
        """Bradley-Terry log-likelihood of comparisons ``Q`` under samples ``u``."""
        value = u @ self.V
        v1 = value[:, Q[:, 0]]
        v2 = value[:, Q[:, 1]]
        t = self.temperature
        logp = t * v1 - torch.log(torch.exp(t * v1) + torch.exp(t * v2))
        return torch.sum(logp, dim=1)

    def dirichletlogpdf(self, x, alpha):
        """Log pdf of Dirichlet(alpha) evaluated at ``x``."""
        return (
            torch.lgamma(torch.sum(alpha))
            - torch.sum(torch.lgamma(alpha))
            + torch.sum((alpha - 1) * torch.log(x), dim=1)
        )

    def random_gradient(self, phi):
        """One stochastic-gradient step on the ELBO wrt ``phi``.

        Uses the reparameterization trick: ``u = softmax(mu + sigma * eps)``
        with ``eps ~ N(0, I_K)``. Returns the gradient and current ELBO
        estimate.
        """
        sample_epsilon = torch.randn(self.W_g, self.K, device=self.device)

        phi = phi.detach().clone().requires_grad_(True)
        mu = torch.log(phi ** 2) - torch.sum(torch.log(phi ** 2)) / self.K
        sigma = 1 / (phi ** 2) * (1 - 2 / self.K) + 1 / (self.K ** 2) * torch.sum(1 / (phi ** 2))
        sigma_sqrt = torch.sqrt(sigma)

        u = mu + sigma_sqrt * sample_epsilon
        sample_u = torch.softmax(u, dim=1)

        term1 = self.loglikelihood(sample_u, self.Q)
        term2 = self.dirichletlogpdf(sample_u, self.alpha)
        term3 = self.dirichletlogpdf(sample_u, phi ** 2)

        loss = (term1 + term2 - term3).mean()
        loss.backward()
        return phi.grad, loss

    def sga(self):
        """Run stochastic gradient ascent with Adam. Returns the trajectories
        of ``phi``, gradient, and ELBO over iterations."""
        phi_set, grad_set, L_set = [], [], []

        # Adam hyperparameters (match the paper: eta=0.01, beta1=0.9,
        # beta2=0.999, epsilon=10e-8 which is 1e-7 by Python scientific notation)
        lr = 0.01
        beta1, beta2 = 0.9, 0.999
        eps = 1e-7

        phi = torch.ones(self.K).to(self.device)
        phi_set.append(phi)

        m = torch.zeros(self.K).to(self.device)
        v = torch.zeros(self.K).to(self.device)

        for t in range(1, self.T):
            grad, L = self.random_gradient(phi)
            L_set.append(L)
            grad_set.append(grad)

            if torch.linalg.norm(grad) < self.e:
                break

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            phi_old = phi
            phi = phi + lr * m_hat / (torch.sqrt(v_hat) + eps)
            phi_set.append(phi)

            if torch.linalg.norm(phi - phi_old) < self.e:
                break

        return torch.stack(phi_set), torch.stack(grad_set), torch.stack(L_set)


class Disaggregation:
    """Lightweight ``SGVI``-without-inference. Computes only the piecewise-
    linear feature matrix ``V``; used when callers need ``V`` but do not
    perform variational updates (e.g., first-round sampling from the prior,
    SOR feasibility construction)."""

    def __init__(self, evalmatrix, intervals_num, device):
        self.device = device
        self.evalmatrix = evalmatrix
        self.n = evalmatrix.shape[0]
        self.m = evalmatrix.shape[1]
        self.intervals_num = intervals_num
        self.K = self.m * self.intervals_num
        self.G = self.criteria_div()
        self.V = self.v_compute()

    def criteria_div(self):
        criteria = torch.sort(self.evalmatrix.clone(), dim=0)[0]
        step = (criteria[-1, :] - criteria[0, :]) / self.intervals_num
        G = torch.zeros((self.intervals_num + 1, self.m)).to(self.device)
        for i in range(self.intervals_num + 1):
            G[i, :] = criteria[0, :] + i * step
        return G

    def v_compute(self):
        V = torch.zeros((self.K, self.n)).to(self.device)
        for i in range(self.n):
            ind = 0
            for j in range(self.m):
                for k in range(self.intervals_num):
                    if self.evalmatrix[i][j] > self.G[k][j] and self.evalmatrix[i][j] >= self.G[k + 1][j]:
                        V[ind, i] = 1
                    elif self.G[k][j] <= self.evalmatrix[i][j] <= self.G[k + 1][j]:
                        V[ind, i] = (self.evalmatrix[i][j] - self.G[k][j]) / (self.G[k + 1][j] - self.G[k][j])
                    ind += 1
        return V
