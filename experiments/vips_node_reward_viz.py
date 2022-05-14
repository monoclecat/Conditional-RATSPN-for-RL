import numpy as np
import torch.distributions as dist
import torch as th
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Gaussians in linspace -10 to 10, 2D
    dist1 = dist.MultivariateNormal(th.as_tensor([-2.0, -2.0]), th.as_tensor([[1.0, 0.0], [0.0, 1.0]]))
    dist2 = dist.MultivariateNormal(th.as_tensor([2.0, 2.0]), th.as_tensor([[2.0, 1.0], [1.0, 2.0]]))


    weights = th.as_tensor([0.5, 0.5])

    # where to evaluate the densities
    n_steps = 501
    x = th.linspace(-5, 5, n_steps)
    grid = th.stack(th.meshgrid(x, x, indexing='xy'), dim=-1)
    # calculate density and apply mixture weights
    c1 = dist1.log_prob(grid).exp() * weights[0]
    c2 = dist2.log_prob(grid).exp() * weights[1]

    result = c1 + c2

    line_at = 200
    pdf_at_line = result[:, line_at]

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(result)
    ax1.axvline(x=line_at, color='r')
    ax2.plot(pdf_at_line)
    ax2.set_title(f"p(x_O, x_I = {round(-5 + line_at/n_steps * 10, 2)})")
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()
    print(1)
