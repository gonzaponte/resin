import sys
import argparse

from collections import defaultdict
from path import Path
from time import time

import numpy             as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Subset
from torch.utils.data import DataLoader

from dataset import DS
from arch    import *

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-folder" , type=Path   , help="folder containing input files")
parser.add_argument("-a", "--arch"         , type=to_arch, help="NN architecture")
parser.add_argument("-r", "--lr"           , type=float  , help="Learning rate"                , default = 1e-4)
parser.add_argument("-e", "--epochs"       , type=int    , help="Learning rate"                , default = 20  )
parser.add_argument("-m", "--max-files"    , type=int    , help="Maximum files to process"     , default = None)
parser.add_argument("-s", "--seed"         , type=int    , help="torch seed"                   , default =  0  )
parser.add_argument("-o", "--output-folder", type=Path   , help="output folder"                , default = "test")

args = parser.parse_args(sys.argv[1:])

if not args.output_folder.exists():
    args.output_folder.mkdir()

f_train = 80
f_valid = 10
assert f_train + f_valid < 100

ds = DS(args.input_folder, args.max_files)

batch_size = ds.n_per_file

idx_train, idx_valid, idx_test = ds.indices(f_train, f_valid)

ds_train = Subset(ds, idx_train); print(f"Training   data: {len(ds_train):>6d}")
ds_valid = Subset(ds, idx_valid); print(f"Validation data: {len(ds_valid):>6d}")
ds_test  = Subset(ds, idx_test ); print(f"Testing    data: {len(ds_test) :>6d}")

loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
loader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False)
loader_test  = DataLoader(ds_test , batch_size=batch_size, shuffle=False)

#########################################################################
# training

if args.seed:
    torch.manual_seed(args.seed)

print("seed =", torch.initial_seed())
nsipms_side = 16
nsipms      = nsipms_side**2
lr          = args.lr
nepochs     = args.epochs
NN          = args.arch
model       = NN(nsipms, ds.n_outputs)
lossf       = nn.MSELoss()
optimizer   = optim.Adam(model.parameters(), lr=lr)
losses      = defaultdict(list)

for epoch in range(1, 1+nepochs):
    t0 = time()
    model.train() # inform the model that we are training
    optimizer.zero_grad()
    for i, (pos, response) in enumerate(loader_train, start=1):
        prediction = model(response)
        loss       = lossf(prediction, pos)
        loss     .backward()
        optimizer.step()

        losses[epoch].append(loss.data.item())

    validation_loss = []
    with torch.no_grad():
        model.eval() # tell the model that we are evaluating
        for i, (pos, response) in enumerate(loader_valid, start=1):
            prediction = model(response)
            loss       = lossf(prediction, pos)
            validation_loss.append(loss.data.item())

    dt = time() - t0
    print(f"Epoch [{epoch}/{nepochs}], Loss: {losses[epoch][-1]:.4f}, Val loss: {np.mean(validation_loss):.1f}, LR: {lr : .4f} DT : {dt:.1f}")

#########################################################################
# testing

plt.figure()
for ie, ep in sorted(losses.items()):
    plt.plot(np.arange(len(ep))+ie*len(ep), ep)

plt.savefig(args.output_folder / "loss.png")


dxs, dys, dps, ps = [[] for i in range(4)]
with torch.no_grad():
    model.eval()
    for pos, response in loader_test:
        predicted = model(response)
        dx, dy    = (predicted - pos).numpy().T
        dxs.extend(dx)
        dys.extend(dy)
        dps.extend((dx**2 + dy**2)**0.5)
        ps.extend((pos**2).sum(axis=1)**0.5)

bins_dxy = np.linspace(-2, 2, 101)
bins_dr  = np.linspace( 0, 2, 101)
normhist = lambda x, b: plt.hist(x, b, weights=np.full(len(x), 100/len(x)))

plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 1); normhist(dxs, bins_dxy); plt.xlabel("dx (mm)"); plt.ylabel("Fraction of events (%)")
plt.subplot(2, 2, 2); normhist(dys, bins_dxy); plt.xlabel("dy (mm)"); plt.ylabel("Fraction of events (%)")
plt.subplot(2, 2, 3); normhist(dps, bins_dr ); plt.xlabel(r"d$\rho$ (mm)"); plt.ylabel("Fraction of events (%)")
plt.subplot(2, 2, 4); plt.hist2d(dxs, dys, (bins_dxy,)*2, cmin=1); plt.xlabel("x (mm)"); plt.ylabel("y (mm)")

plt.tight_layout()
plt.savefig(args.output_folder / "deltas.png")

plt.figure()
bins_r = np.linspace(0, bins_dxy.max(), 101)
plt.hist2d(ps, dps, (bins_r, bins_dr), cmin=1)
plt.xlabel(r"$\rho$ (mm)")
plt.ylabel(r"d$\rho$ (mm)")
plt.savefig(args.output_folder / "deltap_p.png")


torch.save(model, args.output_folder / "model.nn")
