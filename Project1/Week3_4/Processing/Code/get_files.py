#This Script is not changed and taken from Mathias Lecture Notes !!
#I went over line by line though and commented it out

#The code opens the clergy data base from the english church
#it iterates over all english regions and saves the website as a html file in the Input folder
#To increase the speed we define a function and feed that to a multiprocessing algorithm


#Import Packages
import requests
import multiprocessing

#define function to save all html files
def get_pages(begin, end):
    #iterate over all different websites (locations in England)
    for loc_id in range(begin, end):
        #feed it the URL
        url= "https://theclergydatabase.org.uk/jsp/locations/DisplayLocation.jsp?locKey="+str(loc_id)

        page=requests.get(url)
        #if page exists print number to track progress of loop and save html file in Input folder
        if page.status_code==200:
            print(loc_id)
            with open("../Input/file"+str(loc_id)+".html" , "wb+") as f:
                f.write(page.content)
        #if page does not exist, print Unhappy + Number in the terminal
        else:
            print("unhappy" + str(loc_id))


#now we feed that function to multiprocessing so it can run tasks at the same time in different intervals of "loc_id"
if __name__ == "__main__":
    #each process runs our previously defined function on one subspace of the "loc_id" range we want to cover
    p1 = multiprocessing.Process(target=get_pages, args=(2, 5000))
    p2 = multiprocessing.Process(target=get_pages, args=(5000, 10000))
    p3 = multiprocessing.Process(target=get_pages, args=(10000, 15000))
    p4 = multiprocessing.Process(target=get_pages, args=(15000, 25000))
    p5 = multiprocessing.Process(target=get_pages, args=(200000, 250000))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()