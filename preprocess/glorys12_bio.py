import copernicusmarine
import os
import pandas as pd

date_list = pd.date_range(start="20230101", end="20230102", freq="D")

for i in range(len(date_list)-1):
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m",
        variables=["dissic", "ph", "talk"],
        username="mafzhang@ust.hk",
        password="zF573853",
        minimum_longitude=-180,
        maximum_longitude=179.75,
        minimum_latitude=-80,
        maximum_latitude=90,
        start_datetime="2023-{:02d}-{:02d}T00:00:00".format(date_list[i].month, date_list[i].day),
        end_datetime="2023-{:02d}-{:02d}T23:59:59".format(date_list[i].month, date_list[i].day),
        minimum_depth=0.49402499198913574,
        maximum_depth=5727.9169921875,
        output_directory="/home/mafzhang/data/GLORYS12_bio/2023-bak/",
        output_filename="Glorys12-carbon-{:02d}-{:02d}.nc".format(date_list[i].month, date_list[i].day)
    )

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
        variables=["fe", "no3", "po4", "si"],
        username="mafzhang@ust.hk",
        password="zF573853",
        minimum_longitude=-180,
        maximum_longitude=179.75,
        minimum_latitude=-80,
        maximum_latitude=90,
        start_datetime="2023-{:02d}-{:02d}T00:00:00".format(date_list[i].month, date_list[i].day),
        end_datetime="2023-{:02d}-{:02d}T23:59:59".format(date_list[i].month, date_list[i].day),
        minimum_depth=0.49402499198913574,
        maximum_depth=5727.9169921875,
        output_directory="/home/mafzhang/data/GLORYS12_bio/2023-bak/",
        output_filename="Glorys12-nutrients-{:02d}-{:02d}.nc".format(date_list[i].month, date_list[i].day)
        )

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
        variables=["chl", "phyc"],
        username="mafzhang@ust.hk",
        password="zF573853",
        minimum_longitude=-180,
        maximum_longitude=179.75,
        minimum_latitude=-80,
        maximum_latitude=90,
        start_datetime="2023-{:02d}-{:02d}T00:00:00".format(date_list[i].month, date_list[i].day),
        end_datetime="2023-{:02d}-{:02d}T23:59:59".format(date_list[i].month, date_list[i].day),
        minimum_depth=0.49402499198913574,
        maximum_depth=5727.9169921875,
        output_directory="/home/mafzhang/data/GLORYS12_bio/2023-bak/",
        output_filename="Glorys12-Phyto-{:02d}-{:02d}.nc".format(date_list[i].month, date_list[i].day)
        )

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
        variables=["nppv", "o2"],
        username="mafzhang@ust.hk",
        password="zF573853",
        minimum_longitude=-180,
        maximum_longitude=179.75,
        minimum_latitude=-80,
        maximum_latitude=90,
        start_datetime="2023-{:02d}-{:02d}T00:00:00".format(date_list[i].month, date_list[i].day),
        end_datetime="2023-{:02d}-{:02d}T23:59:59".format(date_list[i].month, date_list[i].day),
        minimum_depth=0.49402499198913574,
        maximum_depth=5727.9169921875,
        output_directory="/home/mafzhang/data/GLORYS12_bio/2023-bak/",
        output_filename="Glorys12-o2-{:02d}-{:02d}.nc".format(date_list[i].month, date_list[i].day)
        )




