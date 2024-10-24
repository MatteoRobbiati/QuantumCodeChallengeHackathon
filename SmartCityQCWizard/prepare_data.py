import pandas as pd


zones = [f"{i:03}" for i in range(1, 13, 1)]

datetime = "2024-08-01 23:45:00"

def zones_data_by_datetime(df, datetime):
    filtered_df = df[df["datetime"] == datetime]
    zones_data = {}
    for zone in zones:
        value = filtered_df.loc[filtered_df["areaAnalisi"] == f"Cagliari - {zone}", "value"].values
        # shifting the label from "003" to "002" 
        new_zone_number = int(zone) - 1
        zones_data.update({f"{new_zone_number:03}" : value[0]})
    return zones_data

