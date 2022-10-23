import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# create data paths
#CHANGE TO YOUR PATH HERE
#data_path needs to lead to directory where the final CSV after webscraping the clergy data base is
data_path = "C:/Users/nikil/OneDrive/Desktop/Uni/M.Sc/Data Analytics/Project1/Week3_4/Processing/Output/"
#fig_path is the empty folder where plots will be saved
fig_path = "C:/Users/nikil/OneDrive/Desktop/Uni/M.Sc/Data Analytics/Project1/Week3_4/Processing/Maps_Figures/"

# read webscraped church data set
#only take these columns
cols = ['cced_id', 'Year', 'Type', 'Office', 'PersonID']
church = pd.read_csv(data_path+'CSV_fin.csv')
church = church[cols]
church.rename(columns={"Year": "year", "Type": "type", "Office":"office"}, inplace = True)

# read population data
pop = pd.read_stata(data_path+'Population_year.dta')

#clean the data

# this function takes a given year and returns the closest year divisble by 25
def nearest_year(year):
    if year%25>12:
        return year - year%25 + 25
    else:
        return year - year%25

# apply nearest_year function (see above)to the year column:
pop['year_25'] = pop['year'].apply(lambda x: nearest_year(x))
# get rid of years outside of interval 1525 - 1850:
pop = pop[(pop['year_25'] <= 1850)&(pop['year_25'] >= 1525)].reset_index(drop = True)

# apply nearest_year function (see above)to the year column:
church['year_25'] = church['year'].apply(lambda x: nearest_year(x))
# get rid of years outside of interval 1525 - 1850:
church = church[(church['year_25'] <= 1850)&(church['year_25'] >= 1525)].reset_index(drop = True)

# take only columns we need for merging pop and church data frames:
pop = pop[['cced_id', 'C_ID', 'latitude', 'longitude', 'Population', 'year_25']]
church = church[['cced_id', 'type', 'PersonID', 'year_25']]

##
#MERGE the two datasets

# merge pop data into church on year and the cced id
df = church.merge(pop, on = ['year_25','cced_id'], how='left')
# delete all obsverations with visitors (type = Libc)
df = df[df['type'] != 'Libc'].reset_index(drop = True)

# For some rows there are missing values in the city column (C_ID).
# We can observe the cced in these periods and use that information to replace the nans in the C_IDs...
# ...that match said cceds in other years.

# Drop Nas
cced_df = pop.dropna(subset=['cced_id', 'C_ID', 'latitude', 'longitude'])
# Create dictionary that matches cceds to C_ID values
cced_to_cid = pd.Series(cced_df.C_ID.values, index = cced_df.cced_id).to_dict()
#Same kind of dictionary for latitude and longitude ( also to replace NAs)
cced_to_lat = pd.Series(cced_df.latitude.values, index = cced_df.cced_id).to_dict()
cced_to_lon = pd.Series(cced_df.longitude.values, index = cced_df.cced_id).to_dict()
del cced_df

# Apply the dictionaries to fill in the NAs in C_ID, longitude, and latitude.
# If we can't match we replace the values with -1
df['C_ID'] = df['cced_id'].apply(lambda x: cced_to_cid[x] if x in cced_to_cid.keys() else -1)
df['latitude'] = df['cced_id'].apply(lambda x: cced_to_lat[x] if x in cced_to_lat.keys() else -1)
df['longitude'] = df['cced_id'].apply(lambda x: cced_to_lon[x] if x in cced_to_lon.keys() else -1)

# turn everything to strings so we can create a unique identifier by adding the values up
df['PersonID'] = df['PersonID'].apply(lambda x: str(x))
df['year_25'] = df['year_25'].apply(lambda x: str(x))
df['C_ID'] = df['C_ID'].apply(lambda x: str(int(x)))
df['PersonID+year_25+C_ID'] = df['PersonID'] + df['year_25'] +df['C_ID']

# Drop duplicate rows, so that we only count a person once in every year/ city
df.drop_duplicates(subset = ['PersonID+year_25+C_ID'], inplace = True)
#drop the column
df = df.drop(columns = ['PersonID+year_25+C_ID'])
# drop the -1 values
df = df[df['C_ID'] != '-1']
df.reset_index(inplace = True, drop = True)



#Goal: Plot the aggregated timeseries of appointments vs population to graphically inspect wether they have been moving together over time

#define a column number = 1 for all rows
df['number'] = 1
#Extract year, city and number variable to Count the amount
#of rows per year x city by adding up the number column
a = df[['year_25', 'C_ID','number']].groupby(by = ['year_25', 'C_ID']).sum().reset_index()

#Now merge sum of appointments back in old data set
df = df.merge(a, how = 'right', on = ['year_25', 'C_ID'])
#drop columns we dont need anymore
df.drop(columns = ['type', 'PersonID', 'cced_id', 'number_x'], inplace = True)
df.rename(columns={'number_y':'appointments'}, inplace = True)

df = df.drop_duplicates()

#In the pop data set, NA's were encoded as zeros for population
#Replace these zeros with NA's
df.replace(0, np.nan, inplace = True)
df.reset_index(drop = True, inplace = True)


#Goal: Plot the aggregated timeseries of appointments vs population to graphically inspect whether they have been moving together over time

#Drop all rows where we dont have population values
#get the appointments per year
data_year=df.dropna()[["year_25", "appointments"]].groupby(by = ["year_25"]).sum()
data_year.reset_index(inplace=True)

#Now: get population per year
data_pop=df.dropna()[["year_25", "Population"]]
data_pop.drop_duplicates(inplace=True)
data_pop=data_pop[["year_25", "Population"]].groupby(by= ["year_25"]).sum().reset_index()

#Merge Population per year and Appointments per year
data_plot=data_year.merge(data_pop, on=["year_25"])
#Lets Plot them
fig,ax=plt.subplots( figsize= (12,6))
ax.plot(data_plot.year_25, data_plot.appointments, marker ="o")
ax.set_xlabel("Year", fontsize=14)
ax.plot(data_plot.year_25, data_plot.Population, marker ="o")
plt.show()

#The levels are so different that we need two different scales
#This way we dont see any variation in Appointments


## Same plot with two different Y axis
#Lets start with Appointments
fig,ax= plt.subplots(figsize=(12,6))
ax.plot(data_plot.year_25, data_plot.appointments, marker ="o", color="red")
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Appointments", color= "red", fontsize= 14)

#now add second axis
ax2=ax.twinx()
ax2.plot(data_plot.year_25, data_plot.Population, marker ="o", color="blue")
ax2.set_ylabel("Population", color= "blue", fontsize= 14)

plt.show()

#Save in Figure Folder:
fig.savefig(fig_path + 'Appoint_vs_Pop_TimeSeries.jpg',
            format='jpeg',)

#Save final data for Prediction Models
print(df.head())
df.to_csv(data_path + 'data_for_pred.csv', header = True)