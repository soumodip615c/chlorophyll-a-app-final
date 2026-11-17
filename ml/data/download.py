"""Download monthly chlorophyll-a (4 km) for the northern Bay of Bengal.

Source: Copernicus Marine, Global Ocean Colour multi-sensor L4 monthly.
Usage:
    copernicusmarine login          # once, free account
    python ml/data/download.py
"""
import argparse
from pathlib import Path

import copernicusmarine as cm

DATASET_ID = "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw")
    p.add_argument("--start", default="1997-09-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--lon", nargs=2, type=float, default=[86.0, 92.0])
    p.add_argument("--lat", nargs=2, type=float, default=[18.0, 23.0])
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cm.subset(
        dataset_id=DATASET_ID,
        variables=["CHL"],
        minimum_longitude=args.lon[0],
        maximum_longitude=args.lon[1],
        minimum_latitude=args.lat[0],
        maximum_latitude=args.lat[1],
        start_datetime=args.start,
        end_datetime=args.end,
        output_directory=str(out),
        output_filename="chl_monthly_bob.nc",
    )
    print(f"Saved to {out / 'chl_monthly_bob.nc'}")


if __name__ == "__main__":
    main()
