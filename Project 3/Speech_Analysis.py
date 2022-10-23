# Packages:
import gc
import os
import re


import joblib
import pandas as pd
import openpyxl
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer



#----------------------------------------------------------------------------------------

file=("C:/Users/nikil/OneDrive/Desktop/Uni/M.Sc/Data Analytics/Project 3/Data/Final.Data.csv")
relevantcolumn = 'speechtext'

# Define %Sample to use
prob = 0.5

# Clusters:
cluster_groups = 4

# How many ngrams?
ngrams = 2

# List of banned words
bannedwords = ['canada', 'minister','people', 'hon', "friend", "speaker", "mr", "government", "men", "man",
               "boy", "year", "years", "states" , "house" , "party", "department", "committee", "province", "member",
               "members", "question", "order", "day"]

# minimum length of speeches
lengthspeech = 1000

# Numbers:
numbers = r'[0-9]'

# Models:
model_path = 'Output/NLTK_Model_'+str(cluster_groups)+'.sav'    ## model
vectorizer_path = 'Output/NLTK_vectorizer_'+str(cluster_groups)+'.sav'    ## vectorizer
class_path = 'Output/NLTK_Clusters_'+str(cluster_groups)+'.csv'    ## model

# Model parameters
token = RegexpTokenizer('[a-zA-Z0-9]+')
vectorizer = TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,ngrams), tokenizer=token.tokenize)


#Define Stem Functions
porter= PorterStemmer()
landcaster = LancasterStemmer()


#CLEAN THIS FUNCTION
# Define a function that reads the data:
def load_data(file,wordcount):
    #arguments: (the file to load, the minimum amount of words each speech should have
    #load data and drop NA
    df = pd.read_csv(file)
    df = df[~df['speakeroldname'].isnull()]
    df = df[~df[relevantcolumn].isnull()]
    #extract year
    s = pd.Series(df['speechdate'])
    s2 = s.str.split(r"-",n=-1, expand=True)
    year = s2.iloc[:,0]
    year = pd.DataFrame(year)
    year.columns = ['year']
    #merge year into data
    document = pd.merge(df,year, left_index=True, right_index=True)
    document = document[['basepk',relevantcolumn,'year']]
    document = document.dropna()
    #drop duplicate rows
    document = document.drop_duplicates(subset=[relevantcolumn])
    #get speech length and drop everything that has less words than definded in wordcount (arg of function)
    document["speechlength"] = document[relevantcolumn].str.len()
    document = document.loc[document["speechlength"]>wordcount]
    gc.collect()
    del df
    return document


def clean_string(text, wordlist):
    #arguments (speech text and banned words)
    #tokenize -> split speech text into list of single words
    tokenized_word = word_tokenize(text)
    #take set of stopwords like (i , me , you ...) from package
    stop_words = set(stopwords.words("english"))
    filtered_list = []
    #Get all words that are not stop word or banned words and append them to filtered list
    for w in tokenized_word:
        if w not in stop_words and w not in wordlist:
            filtered_list.append(w)

    #filtered_list contains list with lots of individual strings

    #define word types we want to keep
    typeselect=['NOUN']
    #assigns to each word the word type and convert that into data frame
    #words is a list of touples (word, wordtype)
    words = nltk.pos_tag(filtered_list,tagset='universal')
    w = pd.DataFrame(words)
    w.columns =['word','type']
    #only take rows with words of certain word type
    rslt_df = w.loc[w['type'].isin(typeselect)]
    #ony take the word columns
    h = list(rslt_df['word'])
    # next line gets rid of individual strings and binds them together to one big string

    for word in h:
        porter.stem(word)
        landcaster.stem(word)
    sentence = ' '.join(h)
    return sentence

### Pre-processing
#load data
data = load_data(file,lengthspeech)
#take out speeches and convert it to data frame
doc = data[relevantcolumn]
smp = pd.DataFrame(doc)
#take subset for computational reasons
smp= smp.sample(n=int(prob*len(smp)), random_state=1)
lendoc = len(smp)

#
for i in range(0,lendoc,1):
    #next three lines show us progress of the loop
    if i % 250 == 0:
        p = i/lendoc
        print('{:2.2%}'.format(p))
    #iterate over all speeches
    #take out speech number i and convert it to small letters
    smp.iat[i, 0] = smp.iat[i, 0].lower()
    #replace every number in the text with nothing
    smp.iat[i, 0] = re.sub(numbers,"",smp.iat[i, 0])
    #apply clean string function previously defined
    smp.iat[i, 0] = clean_string(smp.iat[i, 0],bannedwords)


print("Training Models")
#Convert text data in numerical data for statistical model
#vectorizer defined in line 50
tfidf = vectorizer.fit(smp[relevantcolumn])
# save vectorizer
joblib.dump(tfidf,vectorizer_path)
#transfrom output in matrix of numarical values
text_counts = vectorizer.fit_transform(smp[relevantcolumn])

## NOW DATA IS READY AND WE CAN APPLY CLUSTERING ALGORITHM

#CLUSTERING
# Training step
#define how many clusters you want (4)
true_k = cluster_groups
#define clustering algorithm
model = KMeans(n_clusters=true_k,init='k-means++', max_iter=300,n_init=10)
#run it on the data
model.fit(text_counts)
#save outfit
joblib.dump(model,model_path)

### What are the groups?
#get the coordinates of the center of each cluster and sort it
order_centroids = model.cluster_centers_.argsort()[:,::-1]

terms = vectorizer.get_feature_names()


#Lets get 20 most important words for each cluster
df=pd.DataFrame()
for i in range(0,true_k):
    wlist =[]
    #for each cluster, get 20 most important words
    for ind in order_centroids[i, :20]:
        #actually print out the word
        print('%s' % terms[ind])
        #append them to th elist
        wlist.append(terms[ind])
    #convert it to data frame
    df_app = pd.DataFrame(wlist)
    #add to final the result of each iteration
    final = pd.concat([df,df_app],axis=1)
    df = final

#rename columns with cluster groups
lst = list(range(0,true_k))
df.columns=lst
#save data
df.to_csv(class_path, index=False)

# So far we have used the speech data to come up with 5 clusters


# Predicting:
print("Opening Model")
vectorizer = joblib.load(open(vectorizer_path,'rb'))
model = joblib.load(open(model_path,'rb'))

# Preprocessing


### Pre-processing
data = load_data(file,lengthspeech)

doc = data[relevantcolumn]
smp = pd.DataFrame(doc)
lendoc = len(smp)

for i in range(0,lendoc,1):
    if i % 250 == 0:
        p = i/lendoc
        print('{:2.2%}'.format(p))
    smp.iat[i, 0] = smp.iat[i, 0].lower()
    smp.iat[i, 0] = re.sub(numbers,"",smp.iat[i, 0])
    smp.iat[i, 0] = clean_string(smp.iat[i, 0],bannedwords)


text_counts = vectorizer.transform(document[relevantcolumn])
data["prediction"] = model.predict(text_counts)
#
#how are speaches distrubuted among clusters
print(data["prediction"].value_counts(normalize=True))

#lets split this up over time
#how much % of speeches fall in each cluster in a given year

# This is where mathias' code ends
#Preparing the data for plotting

#take out only year and prediction
dat_plot=data[["year", "prediction"]]
dat_plot.rename(columns={"year":"year", "prediction":"pred"}, inplace=True )


#Get prediction seperated by every year
by_year=dat_plot.groupby(by="year")
dat_plot=by_year["pred"].value_counts()
dat_plot=dat_plot.unstack(level=0)
dat_plot.fillna(0, inplace=True)

#Get share for each group in each year
plot_df=pd.DataFrame()
for year in dat_plot.columns:
    plot_df[f"{year}"]=dat_plot[year]/dat_plot[year].sum()

#reshape from wide to long
plot_df.reset_index(inplace=True)
plot_df=plot_df.melt(id_vars="pred")
plot_df.rename(columns={"pred":"pred","variable":"year", "value": "share" }, inplace=True)

print(plot_df.head())

#Plot the share of speaches going into each cluster over time
print(plot_df.groupby("pred")["share"].plot())

#look at the 20 most important words in each cluster to understand what they are about
clusters=pd.read_csv("C:/Users/nikil/OneDrive/Desktop/Uni/M.Sc/Data Analytics/Project 3/Output/NLTK_Clusters_5.csv")
print(clusters)
