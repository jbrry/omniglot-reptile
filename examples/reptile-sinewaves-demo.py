# Source: https://gist.githubusercontent.com/joschu/f503500cda64f2ce87c8288906b09e2d/raw/28ef66e0b072cb2db6fc017d2b9e4798b5cbe311/reptile-sinewaves-demo.py

import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy

seed = 0
plot = True
inner_step_size = 0.02 # stepsize in inner SGD
inner_epochs = 1 # number of epochs of each inner SGD
outer_step_size0 = 0.1 # stepsize of outer optimization, i.e., meta-optimization
n_iterations = 30000 # number of outer updates; each iteration we sample one task and update on it # Q: how many tasks?

rng = np.random.RandomState(seed)
torch.manual_seed(seed)

# Define task distribution
x_all = np.linspace(-5, 5, 50)[:, None] # All of the x points
n_train = 10 # Size of training minibatches
def gen_task():
    "Generate classification problem"
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x : np.sin(x + phase) * ampl
    return f_randomsine

# Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)

def to_torch(x):
    return ag.Variable(torch.Tensor(x))

def train_on_batch(x, y):
    x = to_torch(x)
    y = to_torch(y)
    model.zero_grad()
    y_pred = model(x)
    loss = (y_pred - y).pow(2).mean()
    loss.backward()
    for param in model.parameters():
        param.data -= inner_step_size * param.grad.data

def predict(x):
    x = to_torch(x)
    return model(x).data.numpy()

# Choose a fixed task and minibatch for visualization
f_plot = gen_task()
xtrain_plot = x_all[rng.choice(len(x_all), size=n_train)]

# Reptile training loop
for iteration in range(n_iterations):
    weights_before = deepcopy(model.state_dict())
    # Generate task
    f = gen_task()
    y_all = f(x_all)
    # Do SGD on this task
    inds = rng.permutation(len(x_all))
    for _ in range(inner_epochs):
        for start in range(0, len(x_all), n_train):
            mb_inds = inds[start:start+n_train]
            train_on_batch(x_all[mb_inds], y_all[mb_inds])
    # Interpolate between current weights and trained weights from this task
    # I.e. (weights_before - weights_after) is the meta-gradient
    # Φ ← Φ + ϵ(W − Φ)
    weights_after = model.state_dict()
    outer_step_size = outer_step_size0 * (1 - iteration / n_iterations) # linear schedule
    model.load_state_dict({name : 
        weights_before[name] + (weights_after[name] - weights_before[name]) * outer_step_size 
        for name in weights_before})

    # Periodically plot the results on a particular task and minibatch
    if plot and iteration==0 or (iteration+1) % 1000 == 0:
        plt.cla()
        f = f_plot
        weights_before = deepcopy(model.state_dict()) # save snapshot before evaluation
        plt.plot(x_all, predict(x_all), label="pred after 0", color=(0,0,1))
        for inner_iter in range(32):
            train_on_batch(xtrain_plot, f(xtrain_plot))
            if (inner_iter+1) % 8 == 0:
                frac = (inner_iter+1) / 32
                plt.plot(x_all, predict(x_all), label="pred after %i"%(inner_iter+1), color=(frac, 0, 1-frac))
        plt.plot(x_all, f(x_all), label="true", color=(0,1,0))
        lossval = np.square(predict(x_all) - f(x_all)).mean()
        plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
        plt.ylim(-4,4)
        plt.legend(loc="lower right")
        plt.pause(0.01)
        model.load_state_dict(weights_before) # restore from snapshot
        print(f"-----------------------------")
        print(f"iteration               {iteration+1}")
        print(f"loss on plotted curve   {lossval:.3f}") # would be better to average loss over a set of examples, but this is optimized for brevity