import copernicusmarine
import os
import pandas as pd

date_list = pd.date_range(start="20230101", end="20230102", freq="D")

for i in range(len(date_list)-1):
    copernicusmarine.subset(
      dataset_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",
      variables=["mlotst", "so", "thetao", "uo", "vo"],
      username="mafzhang@ust.hk",
      password="zF573853",
      minimum_longitude=-180,
      maximum_longitude=179.9166717529297,
      minimum_latitude=-80,
      maximum_latitude=90,
      start_datetime="2023-{:02d}-{:02d}T00:00:00".format(date_list[i].month, date_list[i].day),
      end_datetime="2023-{:02d}-{:02d}T23:59:59".format(date_list[i].month, date_list[i].day),
      minimum_depth=0.49402499198913574,
      maximum_depth=5727.9169921875,
      output_directory="/home/mafzhang/data/GLORYS12/2023-bak/",
      output_filename="Glorys12-{:02d}-{:02d}.nc".format(date_list[i].month, date_list[i].day)
    )

