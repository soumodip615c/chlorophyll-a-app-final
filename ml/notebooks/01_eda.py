# Run cell by cell in VS Code: click "Run Cell" above each "# %%" line
# (or put the cursor in a cell and press Shift+Enter).

# %% 1. Load the data
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("data/raw/chl_monthly_bob.nc")
chl = ds["CHL"]
print("shape (time, lat, lon):", chl.shape)
print("units:", chl.attrs.get("units"))
print("time range:", str(chl.time.values[0])[:10], "to", str(chl.time.values[-1])[:10])

# %% 2. How much data is missing (clouds, land, turbid water)?
frac_nan = chl.isnull().mean(dim=["latitude", "longitude"])
frac_nan.plot(figsize=(12, 3))
plt.title("Fraction of missing pixels per month")
plt.show()

print("Average missing fraction by calendar month (1=Jan ... 12=Dec):")
print(frac_nan.groupby("time.month").mean().values.round(2))

# %% 3. What do the values look like?
vals = chl.values[~np.isnan(chl.values)]
print("percentiles 1/25/50/75/99:", np.percentile(vals, [1, 25, 50, 75, 99]).round(3))
print("max:", vals.max())

fig, ax = plt.subplots(1, 2, figsize=(12, 3))
ax[0].hist(vals, bins=100, range=(0, 20))
ax[0].set_title("CHL (raw, clipped at 20)")
ax[1].hist(np.log10(vals), bins=100)
ax[1].set_title("log10(CHL)")
plt.show()

# %% 4. Seasonal cycle: average map for each calendar month
clim = chl.groupby("time.month").mean("time")
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for m, ax in zip(range(1, 13), axes.ravel()):
    np.log10(clim.sel(month=m)).plot(ax=ax, vmin=-1.5, vmax=1, add_colorbar=False)
    ax.set_title(f"Month {m}")
plt.tight_layout()
plt.show()

# %% 5. Time series at one location (near the Hooghly mouth)
pt = chl.sel(latitude=21.6, longitude=88.0, method="nearest")
print("pixel used:", float(pt.latitude), float(pt.longitude))
pt.plot(figsize=(12, 3))
plt.title("CHL at one pixel over time")
plt.show()
