import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# graphons for simulated data
def graphon_1(x):
    'w(u,v) = u * v'
    p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=device)
    u = p + x.reshape(1, -1)
    v = p + x.reshape(-1, 1)
    graphon = u * v
    return graphon


def graphon_2(x):
    'w(u,v) = exp{-(u^0.7 + v^0.7))}'
    p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=device)
    u = p + x.reshape(1, -1)
    v = p + x.reshape(-1, 1)
    graphon = torch.exp(-(torch.pow(u, 0.7) + torch.pow(v, 0.7)))
    return graphon


def graphon_3(x):
    'w(u,v) = (1/4) * [u^2 + v^2 + u^(1/2) + v^(1/2)]'
    p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=device)
    u = p + x.reshape(1, -1)
    v = p + x.reshape(-1, 1)
    graphon = 0.25 * (torch.pow(u, 2) + torch.pow(v, 2) + torch.pow(u, 0.5) + torch.pow(u, 0.5))
    return graphon


def graphon_4(x):
    'w(u,v) = 0.5 * (u + v)'
    p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=device)
    u = p + x.reshape(1, -1)
    v = p + x.reshape(-1, 1)
    graphon = 0.5 * (u + v)
    return graphon


def graphon_5(x):
    'w(u,v) = 1 / (1 + exp(-10 * (u^2 + v^2)))'
    p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=device)
    u = p + x.reshape(1, -1)
    v = p + x.reshape(-1, 1)
    graphon = 1 / (1 + torch.exp(-10 * (torch.pow(u, 2) + torch.pow(v, 2))))
    return graphon


def graphon_6(x):
    'w(u,v) = |u - v|'
    p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=device)
    u = p + x.reshape(1, -1)
    v = p + x.reshape(-1, 1)
    graphon = torch.abs(u - v)
    return graphon


def graphon_7(x):
    'w(u,v) = 1 / (1 + exp(-(max(u,v)^2 + min(u,v)^4)))'
    p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=device)
    u = p + x.reshape(1, -1)
    v = p + x.reshape(-1, 1)
    graphon = 1 / (1 + torch.exp(-(torch.pow(torch.max(u, v), 2) + torch.pow(torch.min(u, v), 4))))
    return graphon


def graphon_8(x):
    'w(u,v) = exp(-max(u, v)^(3/4))'
    p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=device)
    u = p + x.reshape(1, -1)
    v = p + x.reshape(-1, 1)
    graphon = torch.exp(-torch.pow(torch.max(u, v), 0.75))
    return graphon


def graphon_9(x):
    'w(u,v) = exp(-0.5 * (min(u, v) + u^0.5 + v^0.5))'
    p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=device)
    u = p + x.reshape(1, -1)
    v = p + x.reshape(-1, 1)
    graphon = torch.exp(-0.5 * (torch.min(u, v) + torch.pow(u, 0.5) + torch.pow(v, 0.5)))
    return graphon


def graphon_10(x):
    'w(u,v) = log(1 + 0.5 * max(u, v))'
    p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=device)
    u = p + x.reshape(1, -1)
    v = p + x.reshape(-1, 1)
    graphon = torch.log(1 + 0.5 * torch.max(u, v))
    return graphon