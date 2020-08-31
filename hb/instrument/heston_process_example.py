import QuantLib as ql
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Utility function to pull out spot and vol paths as Pandas dataframes
def generate_multi_paths_df(sequence, num_paths):
    spot_paths = []
    vol_paths = []

    for i in range(num_paths):
        sample_path = seq.next()
        values = sample_path.value()

        spot, vol = values

        spot_paths.append([x for x in spot])
        vol_paths.append([x for x in vol])

    df_spot = pd.DataFrame(spot_paths, columns=[spot.time(x) for x in range(len(spot))])
    df_vol = pd.DataFrame(vol_paths, columns=[spot.time(x) for x in range(len(spot))])

    return df_spot, df_vol

today = ql.Date(1, 7, 2020)
v0 = 0.01; kappa = 1.0; theta = 0.04; rho = -0.3; sigma = 0.4; spot = 100; rate = 0.0

# Set up the flat risk-free curves
riskFreeCurve = ql.FlatForward(today, rate, ql.Actual365Fixed())
flat_ts = ql.YieldTermStructureHandle(riskFreeCurve)
dividend_ts = ql.YieldTermStructureHandle(riskFreeCurve)
spot_quote = ql.SimpleQuote(spot)
spot_handle = ql.QuoteHandle(spot_quote)
heston_process = ql.HestonProcess(flat_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)

timestep = 90
length = 90./360.
times = ql.TimeGrid(length, timestep)
dimension = heston_process.factors()

rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(dimension * timestep, ql.UniformRandomGenerator()))
seq = ql.GaussianMultiPathGenerator(heston_process, list(times), rng, False)

df_spot, df_vol = generate_multi_paths_df(seq, 1)
df_spot.head()