

#NOTE 1: This is not my code but taken from Mathias Lecture Script
#I went over line by line and commented it out
#NOTE 2: You have to change the file path in line 94

#This File takes all html files from "get_files", grabs the data we want and saves it in one large CSV File


# Packages
from os import listdir
from os.path import isfile, join
import pandas as pd
from bs4 import BeautifulSoup
import os

# Get a list of the csv files we want to iterate over:
#do this by creating list object from all html files previously web scraped in the Input Folder
files= [f for f in listdir("../Input/") if isfile(join("../Input/",f))]


# Loop over all files:
for f in files:
    #read in the html file
    with open ("../Input/"+str(f), "rb") as file:
        #read it with bs4 package and create soup object
        soup = BeautifulSoup(file.read(),'html.parser')
    ## Catch the cced_id: The css type is always dfn
    cced = soup.find_all('dfn')
    #if there is no value replace with 0
    if not len(cced):
        cced.extend('0')
    # Catch the top three rows, which contain county, and diocese__jurisdiction/geographic
    top_rows = soup.find_all('ul', class_='s2')
    if len(top_rows):
        top_rows = top_rows[0].find_all('li')
        if len(top_rows) > 2:
            for i in range(0, 3):
                top_rows[i] = str(top_rows[i]).rsplit('</label>')[1].rsplit('</li>')[0]
            top_rows = top_rows[:3]
        else:
            top_rows = ['0','0','0']
    if soup.h1:
        top_rows.extend(soup.h1)
    else:
        top_rows.extend('0')


    table = soup.find('div','tb s2')
    # iterate over all tables


    #object with all tables
    t1 = soup.find_all('table')
    #create empty data frame with columns we want
    df = pd.DataFrame(columns=['Names', 'PersonID', 'Year', 'Type'
        , 'Full', 'parish', 'county', 'diocese__jurisdiction', 'diocese__geographic', 'cced_id'])

    if len(t1) > 1:
        t2 = t1[1]
        for row in t2.tbody.find_all('tr'):

            columns = row.find_all('td') + top_rows + cced
            if len(columns) > 0:
                #extract all variables we want
                #edit / reformat the strings as we want them
                names = columns[0].text.strip().replace("\r", "").replace("\n", "").replace("  ", " ")
                year = columns[1].text.strip().replace("\r", "").replace("\n", "").replace("  ", " ")
                type = columns[2].text.strip().replace("\r", "").replace("\n", "").replace("  ", " ")
                office = columns[3].text.strip().replace("\r", "").replace("\n", "").replace("  ", " ")
                full = columns[4].text.strip().replace("\r", "").replace("\n", "").replace("  ", " ")
                c = columns[0].find('a', href=True)
                if c.__str__() != 'None':
                    persid = c['href'].replace('../persons/CreatePersonFrames.jsp?PersonID=', '')
                else:
                    persid = '0'
                county = columns[5].strip().replace("\r", "").replace("\n", "").replace("  ", " ")
                diocese__jurisdiction = columns[6].strip().replace("\r", "").replace("\n", "").replace("  ", " ")
                diocese__geographic = columns[7].strip().replace("\r", "").replace("\n", "").replace("  ", " ")
                parish = columns[8].strip().replace("\r", "").replace("\n", "").replace("  ", " ")
                cced_id = str(columns[9]).strip().replace("<dfn>", "").replace("\r", "").replace("  ", " ").replace(
                    '</dfn>', '').replace('\n', '')
                #append data from each iteration to data frame and put it in the correct collumn
                df = df.append(
                    {'Names': names, 'PersonID': persid, 'Year': year, 'Type': type, 'Office': office, 'Full': full,
                     'parish': parish, 'county': county, 'diocese__jurisdiction': diocese__jurisdiction,
                     'diocese__geographic': diocese__geographic, 'cced_id': cced_id}, ignore_index=True)
        f1 = f.replace('.html','').replace('file','')
        #save data frame for each file as csv in Output Folder
        df.to_csv("../Output/CSV/data"+str(f1)+".csv", index= False)


#CHANGE File Path   HERE
file_path= "C:/Users/nikil/OneDrive/Desktop/Uni/M.Sc/Data Analytics/Project1/Week3_4/Processing/Output/"
os.chdir(file_path+"CSV")


#create final data frame with column names we want
df_fin = pd.DataFrame(columns=['Names', 'PersonID', 'Year', 'Type', 'Full', 'parish', 'county', 'diocese__jurisdiction', 'diocese__geographic', 'cced_id'])


#grab all csv files we just created

csvfiles = [f for f in os.listdir(file_path+"CSV")]

print(csvfiles)
#append them to the final data frame called df_fin
df_fin = pd.concat(map(pd.read_csv, csvfiles), ignore_index= True)


df_fin.Names = df_fin.Names.replace(r'\s+', ' ', regex=True)
df_fin.Type = df_fin.Type.replace(r'\s+', ' ', regex=True)
#save this data frame as final csv data
df_fin.to_csv(file_path+"CSV_fin.csv", index= False)