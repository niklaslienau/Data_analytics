import os
import tqdm
import pandas as pd




#get list of all file names in data folder
all_files=os.listdir("Data")

#create empy data frame to save data in
final_data=pd.DataFrame()

#download each file and append it to data frame called final
for file in all_files:
    print(f"{file}")
    data_file = pd.read_excel(f"Data/{file}")
    final_data = pd.concat([final_data, data_file], ignore_index=True)

final_data.to_csv("Data/Final.Data.csv")

print(final_data.info())


