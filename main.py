#!/usr/bin/env python3

import arbor as A
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from dataclasses import dataclass

# Settings on regions
@dataclass
class parameters:
    cm: float = None # membrane capacitance
    T: float = None  # temperature
    Vm: float = None # initial potential
    rL: float = None # axial resistivity

with open('fit_parameters.json') as fd:
    fit = json.load(fd)

parm = defaultdict(parameters)
mech = defaultdict(dict)
for block in fit['genome']:
    # awkward format, part une: { "section": region, "name": parameter, "value": float, "mechanism": mech }
    # repetitions of section/mechanism are possible to set different parameters
    mch = block['mechanism'] or 'pas' # empty string on mechanism means 'pas' or leak
    reg = block['section']            # dend | apic | soma | axon
    val = float(block['value'])       # value
    key = block['name'].removesuffix(('_' + mch)) # {parameter name}_{mechanism name} unless 'pas' ...
    # ... then it _could_ be one the the parameter thingies ...
    if mch == "pas":
        if key == 'cm':
            parm[reg].cm = val/100.0
        elif key == 'Ra':
            parm[reg].rL = val
        elif key == 'Vm':
            parm[reg].Vm = val
        elif key == 'celsius':
            parm[reg].T = val + 273.15
        continue # (if that's the case, do _not_ try to set something on a mechanism.)
    # ... or just a value on 'pas', but then, there's no suffix. ¯\_(ツ)_/¯
    mech[(reg, mch)][key] = val

# any passive things not set above are going are.
cond = fit['conditions'][0]

# pull out reversal potentials
ions = []
for kv in cond['erev']:
    # part deux: the format {'section': name, 'e<ion_0>': value, 'e<ion_1>': value, ...}
    # makes for slightly awkward parsing
    region = kv['section']
    for k, v in kv.items():
        if k != 'section':
            ions.append((region, k[1:], float(v)))

# Now we can apply the parsed data to our cell.
dec = A.decor()

# find (cell) global defaults
dec.set_property(tempK=float(cond['celsius']) + 273.15,
                 Vm=float(cond['v_init']),
                 cm=None, # not set in fit.json
                 rL=float(fit['passive'][0]['ra']))

# splat on parameters...
for rg, p in parm.items():
    dec.paint(f'"{rg}"', tempK=p.T, Vm=p.Vm, cm=p.cm, rL=p.rL)

# ... reversal potentials, ...
for rg, i, e in ions:
    dec.paint(f'"{rg}"', ion_name=i, rev_pot=e)

# and mechanisms
for (rg, m), vs in mech.items():
    dec.paint(f'"{rg}"', A.density(m, vs))

# add stimulus and spike detector
ctr = '(on-components 0.5 (tag 1))'
dec.place(ctr, A.iclamp(100, 1000, 0.2), 'inj')
dec.place(ctr, A.threshold_detector(-40), 'det')

# Read patched morphology
mrf = A.load_swc_neuron('reconstruction-patched.swc')

# construct a simple simulation
cel = A.cable_cell(mrf, dec, A.label_dict().add_swc_tags())
sim = A.single_cell_model(cel)

# enable all the little mechanisms Allen uses.
sim.properties.catalogue.extend(A.allen_catalogue(), '')

# to extract the soma's potential
sim.probe('voltage', ctr, frequency=20)
sim.run(1200)

# plotting
fg, ax = plt.subplots()
trace = sim.traces[0]
ts = np.array(trace.time[:])
vs = np.array(trace.value[:]) + 14.0 # junction potential shift, see manifest
sp = np.array(sim.spikes)
ax.plot(ts, vs, label='Arbor', zorder=15)
ax.scatter(sp, np.zeros_like(sp) - 40, color='red', zorder=20, label='Spike')
ax.set_xlabel('t/ms')
ax.set_ylabel('U/mV')
plt.savefig("arbor.png", bbox_inches='tight')
plt.savefig("arbor.pdf", bbox_inches='tight')
ax.legend()
