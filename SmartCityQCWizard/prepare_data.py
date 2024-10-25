import pandas as pd


zones = [f"{i:03}" for i in range(0, 13, 1)]

def zones_data_by_datetime(df, datetime):
    filtered_df = df[df["datetime"] == datetime]
    print(filtered_df)
    zones_data = {}
    for zone in zones:
        value = filtered_df.loc[filtered_df["areaAnalisi"] == f"Cagliari - {zone}", "value"].values
        zones_data.update({zone : value[0]})
    return zones_data

