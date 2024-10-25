import pandas as pd 

from prepare_data import save_unique_datetime_list

datasets = [
    "presenza_15_010824-140824",
    "presenza_15_150824-310824",
    "presenza_15_010924-140924",
    "presenza_15_150924_300924",
    "presenza_15_011024-081024",
]

path = "./unique_attendance_15"

for i in range(len(datasets)):
    df = pd.read_csv(f"{path}/{datasets[i]}.csv")
    save_unique_datetime_list(df, f"{path}/unique_dates_{datasets[i]}.txt")