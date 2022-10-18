import matplotlib.pyplot as plt

import colorcet
import flowkit
import flowutils
import numpy as np
import pandas as pd
import pathlib
import sklearn.neighbors

def estimate_kernel_density(data, sampling_range, bandwidth=0.008, **kwargs):
    """
    """
    kde = sklearn.neighbors.KernelDensity(bandwidth=bandwidth, **kwargs)
    kde.fit(data[:, np.newaxis])
    pdf = np.exp(kde.score_samples(sampling_range[:, np.newaxis]))
    pdf[0] = 0
    pdf[-1] = 0
    return pdf

def get_kernel_density(r, fcs_dir, fluor, fluor_name, x_coordinates, logicle_transform):
    """For a given fluor get the estimated kernel density
    Parameters:
    r: a dataframe row (using panda's apply)
    fcs_dir: dir with fcs files
    fluor: the fluor column name from the FCS file, for GFP it is FL1-A
    fluor_name: readable name for the fluor, such as "GFP"
    x_coordinates:
    """
    fcs_file = f"{r.plate:02d}-Well-{r.row}{r.column}.fcs"
    fcs_file = pathlib.Path(fcs_dir, fcs_file)
    
    sample = flowkit.Sample(fcs_file)
    sample.apply_transform(logicle_transform)
    raw_sample_data = sample.as_dataframe(source="xform", col_order=[fluor], col_names=[fluor_name])[fluor_name].values
    ekd = estimate_kernel_density(raw_sample_data, x_coordinates)
    return ekd

def plot_single_strain(tdf, ax, major_xtick_positions, major_tick_labels, minor_xtick_positions):
    """Plot all replicates for a single strain
    """
    # to do
    pass


logicle_transform = flowkit.transforms.LogicleTransform("logicle", param_t=2**24, 
                                                        param_w=.25, param_m=7, param_a=0)

base10_min = 1
base10_max = 11

def sci_format(number):
    if abs(number) <= 10:
        return f"{number:.0f}"
    return f"{number:.0e}"
    #return np.format_float_scientific(number, precision=2, exp_digits=1)

major_tick_positions = list(reversed(-np.logspace(base10_min, base10_max, base10_max - base10_min + 1))) + [-10, 0] + list(np.logspace(base10_min, base10_max, base10_max - base10_min + 1))
transformed_major_tick_positions = flowutils.transforms._logicle(major_tick_positions, t=logicle_transform.param_t, m=logicle_transform.param_m, w=logicle_transform.param_w, a=logicle_transform.param_a)
major_tick_map = dict(zip(transformed_major_tick_positions, map(sci_format, major_tick_positions)))

minor_tick_positions = list(np.linspace(-10, 0, 11))[1:-1] + list(np.linspace(0, 10, 11))[1:-1]
for i in range(base10_min, base10_max):
    minor_tick_positions.extend(list(reversed(-np.linspace(10**i, 10**(i+1), 10)))[1:-1])
for i in range(base10_min, base10_max):
    minor_tick_positions.extend(list(np.linspace(10**i, 10**(i+1), 10))[1:-1])
transformed_minor_tick_positions = flowutils.transforms._logicle(minor_tick_positions, t=logicle_transform.param_t, m=logicle_transform.param_m, w=logicle_transform.param_w, a=logicle_transform.param_a)

major_tick_labels = [major_tick_map[x] for x in transformed_major_tick_positions]

min_x = 0
max_x = 1
extra = 0.4
x_coordinates = np.linspace(min_x-extra, max_x+extra, 1000)

fcs_dir = "2022-06-16-fcs"
plate_map = "plate-map.csv"

df = pd.read_csv(plate_map)

# get the estimated kernel density for GFP
df["ekd"] = df.apply(get_kernel_density, args=(fcs_dir, "FL1-A", "GFP", x_coordinates, logicle_transform), axis=1)

# divide by the overall max from all the samples
df["norm_kd"] = df.ekd / df.ekd.apply(max).max()

strains = df.strain.unique()

fig, axs = plt.subplots(len(strains), 1, figsize=(6, 2*len(strains)), sharex=True)
for i, (ax, strain) in enumerate(zip(fig.axes, strains)):
    tdf = df[df.strain == strain]
    plot_single_strain(tdf, ax, transformed_major_tick_positions, major_tick_labels, 
                       transformed_minor_tick_positions)

ax.set_xlabel("log10 FITC")
fig.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig("plot.png", facecolor="white")
