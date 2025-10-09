from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import jit

import jaxley as jx
from jaxley.channels import Na, K, Leak
from jaxley.synapses import IonotropicSynapse
from jaxley.connect import fully_connect, connect

from Cell import *

# Build the cell.
comp = jx.Compartment()
branch = jx.Branch(comp, ncomp=4)
cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1, 2, 2])

# Insert channels.
cell.insert(Leak())
cell.branch(0).insert(Na())
cell.branch(0).insert(K())
cell.branch(1).insert(Na())
cell.branch(1).insert(K())

# Change parameters.
cell.set("axial_resistivity", 200.0)

# Define a network. `cell` is defined as in previous tutorial.
net = jx.Network([cell for _ in range(100)])

net.compute_xyz()
net.rotate(180)
net.arrange_in_layers(layers=[35, 15, 35, 15], within_layer_offset=40, between_layer_offset=500)

pre = net.cell(range(35))
post = net.cell(range(35, 50))
fully_connect(pre, post, IonotropicSynapse())

pre = net.cell(range(50, 85))
post = net.cell(range(85, 100))
fully_connect(pre, post, IonotropicSynapse())

pre = net.cell(range(35, 50)).branch(3).loc(1.0)
post = net.cell(range(60, 75)).branch(0).loc(0.0)
connect(pre, post, IonotropicSynapse())

pre = net.cell(range(82, 85)).branch(0).loc(1.0)
post = net.cell(range(32, 35)).branch(3).loc(0.0)
connect(pre, post, IonotropicSynapse())

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
_ = net.vis(ax=ax, detail="full")
plt.show()

net.set("IonotropicSynapse_gS", 0.0003)
net.select(edges=[0, 1]).set("IonotropicSynapse_gS", 0.0004)

i_delay = 10.0  # ms
i_amp = 0.05  # nA
i_dur = 10.0  # ms

# Duration and step size.
dt = 0.025  # ms
t_max = 250.0  # ms
time_vec = jnp.arange(0.0, t_max + dt, dt)

net.insert(Na())
net.insert(K())
net.insert(Leak())

net.delete_stimuli()
for stim_ind in range(50):
    i_dur = 10.0 * stim_ind  # ms
    i_amp = 0.03 / (stim_ind+2)
    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    net.cell(stim_ind).branch(0).loc(0.0).stimulate(current)

net.delete_recordings()
for cell_ind in range(50):
    net.cell(cell_ind).branch(1).loc(0.0).record()

s = jx.integrate(net, delta_t=dt)
fig, ax = plt.subplots(1, 1, figsize=(4, 2))
_ = ax.plot(s.T)
plt.show()

