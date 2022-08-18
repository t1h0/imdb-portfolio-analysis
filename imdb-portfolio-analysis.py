"""The goal of this project is to provide a framework to analyse the portfolio of persons active in the movie/tv business by extracting their opuses titles from IMDB and retreiving more relevant info regarding these opuses from OMDB. This info can than be analyzed using the class DataAnalysis, which (for the moment only) inherits a function, analyse_text, to analyse the plot of the opuses. All of the person's opuses' plots are here anaylsed regarding their keywords, extracted using the "term frequency times inverse document frequency (TF-IDF)"-measurement. These keywords therefore represent the words, the person's opuses are most often described with. Their TF-IDF-score can, for example, be put in relation with the number of opuses' plot-descriptions, they're used in. A AnalysisProcessing class accomplishes this examination with the summary()-method. For now, the class InputHandle can be used for a test run."""

import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup as bs
import re
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer as tfidfVect
from sklearn.feature_extraction import text as sklearnText
#import omdbapi.movie_search

class PortfolioRetrieval():
    """Retrieves the portfolio of a person active in the movie/tv business"""
    def __init__(self, API: str):
        self.API = API
        self.base_url_omdb = f"http://www.omdbapi.com/?apikey={self.API}&plot=short&"
        self.base_url_imdb = "https://www.imdb.com"
    
    def get_portfolio(self, person,limit=None):
        """This function manages the process of extracting titles (with IMDB-IDs, -ratings and plot-summarys) of opuses
        associated to a certain person active in the Movie/TV business by using IMDB and OMDB.
        The results are stored in the DataFrame "portfolio".
        Parameters are:
        -person: a string resembling the name of a person active in the Movie/TV business
        -limit: int, indicating if only a certain amount of opuses should be extracted. That is, if limit=10,
        only the first 10 titles (sorted by popularity) listet on the person's portfolio on IMDB are extracted.
        Use this, if you have an OMDB-API-limit."""

        # -------------------------------------
        # I. preparation
        # -------------------------------------
        self.info = {} # info about the person will go in here
        self.name = person.title()
        self.info["Name"] = self.name

        # -------------------------------------
        # II. format the person in question to be used as a query in IMDB's search
        # (removing leading and trailing spaces and replacing spaces in between with "+")
        # -------------------------------------
        query = person.strip().replace(" ","+")

        # -------------------------------------
        # III. extract IMDB-ID
        # -------------------------------------
        self.imdb_id = self.__extract_pers_id(query)
        self.info["imdbID"] = self.imdb_id

        # -------------------------------------
        # IV. extract titles and corresponding IMDB-IDs
        # -------------------------------------
        self.__extract_portfolio(limit)

        # -------------------------------------
        # V. extract year and plot from OMDB
        # -------------------------------------
        self.__complete_portfolio()

    def __extract_pers_id(self,query):
        """extracts the IMDB-ID of a person by taking it from the first result in IMDB search"""

        # -------------------------------------
        # I. get (requests.get) the search result on imdb as html (.text)
        # -------------------------------------
        imdb_search_html = requests.get(f"{self.base_url_imdb}/find?s=nm&q={query}").text

        # -------------------------------------
        # II. convert the html to beautiful-soup-object (bs())
        # III. find the first result with beautiful soup's select_one command
        # --> criteria: find a-tag which includes a href and which directly descends from a tag with "result_text" class
        #               (IMDB's result-list is formatted as ..<td class="result_text"><a href=..)
        # IV. take the link from that extraction
        # -------------------------------------
        imdb_search_href = bs(imdb_search_html).select_one(".result_text>a",href=True)["href"]

        # -------------------------------------
        # V. extract the persons id by stripping the link
        # -------------------------------------
        return imdb_search_href.replace("/name/","").replace("/","")

        # -------------------------------------
        # VI. return the DataFrame "portfolio"
        # -------------------------------------
        return self.portfolio

    def __extract_portfolio(self,limit):
        """extracts the name and IMDB-ID of every opus the person is associated with
        and returns a panda DataFrame with movie titles as rows and the IMDB-ID in a column"""

        # -------------------------------------
        # I. prepare the base url for the portfolio
        # -------------------------------------
        imdb_portfolio_url = f"{self.base_url_imdb}/search/title/?adult=include&view=simple&role={self.imdb_id}&"

        # -------------------------------------
        # II. get the total number of opuses by this person
        # -------------------------------------

        # the total number is displayed on every portfolio on IMDB
        # the reduce loading times, it is taking from an overview with only 1 opus displayed ("count=1") 
        total_op_html = requests.get(f"{imdb_portfolio_url}count=1").text
        # look for the tag which includes the toal number of opuses
        total_op_text = bs(total_op_html).find("span",string=re.compile("1-1 of "))
        # format the string to only hold the total number and convert to int
        self.total_op = int(total_op_text.string.replace("1-1 of ","").replace(" titles.",""))
        self.info["Total"] = self.total_op #assign total number to info dictionary


        # -------------------------------------
        # III. iterate through the pages of the person's portfolio
        # -------------------------------------

        result = [] # results will go in here and be appended to a DataFrame "portfolio" at the end
        
        # -------------------------------------
        # prepare some scraping limits either
        # - because a manual scraping limit is given, or
        # - to reduce loading times and runtime
        # - because auf IMDB's given limits (max count = 250)
        # -------------------------------------
        if limit == 0:
            limit =None
        iterlimit = self.total_op+1  # limit for iterating through pages
        count = 250 # limit for opuses displayed at a single portfolio site on IMDB
        # check if a manual scraping limit is given and if it is below the total amount of opuses of this person
        if isinstance(limit,int) and limit < self.total_op:
            iterlimit = limit+1
            if limit < 250: # if the limit is below 250, we don't have to load more opuses when getting the portfolio
                count = limit

        # "start" defines the index number of the list's first (top) opus to be displayed in IMDB.
        # The limit is either a given scarping limit or the total amount of opuses by this person
        for start in range(1,iterlimit,250):

            # -------------------------------------
            # get the html
            # the "count"-parameter in IMDB's URL defines, how many opuses should be displayed on one site (max.250)
            # if a scraping-limit is given, this is used for the amount of opuses displayed (to reduce loading time)
            # -------------------------------------
            imdb_movies_html = requests.get(f"{imdb_portfolio_url}count={count}&start={start}").text

            # -------------------------------------
            # convert the html text (.text) to beautiful-soup-object (bs())
            # find every title's IMDB-Link with beautiful soup's select command
            # --> criteria: find every a-tag which includes a href and which descends
            #               from a tag with "lister-item-header" class
            #               (IMDB's portfolio site is formatted as ..<[tag] class="lister-item-header">..<a href=..)
            # -------------------------------------
            imdb_movies_tags = bs(imdb_movies_html).select(".lister-item-header a",href=True)

            # -------------------------------------
            # extract the IMDB ids (inside of the hrefs), the opuses' titles (inside of the tags)
            # and the type of opus from the tags
            # -------------------------------------
            
            append = True # True if the opus has to be included in the results-list

            # itereate trough the tags
            for i in imdb_movies_tags:
                # -------------------------------------
                # define the type of opus:
                # "standalone" for autonomous opuses like movies, Tv specials etc
                # "episode" for an episode inside a series
                # criteria: if a sibling of the i-th a-tag has "Episode:" as string (append = false),
                #           the i-th tag is the name of the series and thus the next a tag (i+1)
                #           inside the <tag class="lister-item-header"></tag> will be the opus,
                #           the person is actually associated with
                # -------------------------------------
                opus_type = ("Standalone" if append else "Episode")
                append = True

                # iterate through the i-th tag's siblings (see above for explanation)
                for sibling in i.next_siblings:
                    if (sibling.string == "Episode:"):
                        append = False # indictaing, that the current i-th tag shouldn't be included in the results
                        break
                if append:
                    # -------------------------------------
                    # append the current tag to the results in a list, by:
                    # 1. inserting the imdb-id as first item (by stripping the href, so that the id remains)
                    # 2. inserting the title (i.e. the tag's content) as second item
                    # 3. inserting the opuses' type as third item
                    # 4. inserting three more NaN-items, representing Year,Plot and IMDB-rating of the opus,
                    #    to be inserted via OMDB
                    # -------------------------------------
                    result.append([i["href"].replace("/title/","").replace("/",""),i.string,opus_type,"NaN","NaN","NaN"])
        # append the result list to the DataFrame "portfolio"
        self.portfolio = pd.DataFrame(result,columns=["imdbID","Title","Type","Year","Plot","imdbRating"])

    def __complete_portfolio(self):
        """complete the DataFrame "portfolio" with Year, Plot and IMDB rating of every title via OMDB API"""

        # iterate through the rows (=opuses) of the searched person in the DataFrame
        for i,row in self.portfolio.iterrows():
            # for every opus get the corresponding json from omdb via IMDB id
            omdb_entry = requests.get(f"{self.base_url_omdb}i={row['imdbID']}").json()
            # update the last three rows "Year", "Plot" and "imdbRating" with the info included in the omdb json
            for column in self.portfolio.columns[3:]:
                try:
                    self.portfolio.at[i,column] = omdb_entry[column]
                except KeyError:
                    pass

class DataAnalysis():
    """Analyses the scraped data"""

    def __init__(self,portfolio):
        self.portfolio = portfolio
    
    def analyse_text(self, text_column,mean_mismatch=False,ignore=[]):
        """This function takes the DataFrame passed to the constructor and analyses multiple strings
        in a specified column (in this case: plot descriptions) by extracting keywords using the
        "term frequency times inverse document frequency (TF-IDF)"-measurement.
        The mean of the TF-IDFs are then calculated over the various texts
        and returned in a new Data Frame with the columns "word" and "TF-IDF".
        parameters:
                    - text-column: specifies the column the strings to be analyzed are in
                    - mean_mismatch: if false, strings/plots not containing the specific word
                                     are not included for mean-calculation
                    - ignore: can contain a list of words to be ignored in the analysis-process.
                              A list of typically ignored english words included in scikit-learn
                              and the string "nan" (will appear if a plot or the film itself wasn't found
                              in OMDB's database) is automatically added."""

        # -------------------------------------
        # I. Generating TF-IDFs
        # -------------------------------------

        # prepare list of words to be ignored
        stop_words_list = ["nan"]
        if isinstance(ignore,list):
            stop_words_list.extend(ignore)
        # initialize the tfidf-vectorizer (see scikit-learn doc for further details)
        tfidf_vect = tfidfVect(stop_words=sklearnText.ENGLISH_STOP_WORDS.union(stop_words_list))
        # generating the tfidfs
        tfidf = tfidf_vect.fit_transform(self.portfolio[text_column])
        # converting the tfidf-result to a usable array
        tfidf_array = tfidf.toarray()
        # this list contains all the words, with their indices matching the tfidf-list's indices
        tfidf_words = tfidf_vect.get_feature_names()

        # -------------------------------------
        # II. Calculating the Means
        # -------------------------------------

        # this will contain the resulting list, containing lists with words as strings and the means as floats
        self.tfidf_means = []

        # iterate through the tfidf-list (columns=words,rows=strings(i.e. text/plots))
        for col in range(len(tfidf_array[0])):
            # adding a list to the result list with the word as string and 0 as TF-IDF-value
            self.tfidf_means.append([tfidf_words[col],0,0])
            rowcount = 0
            totalrows = 0
            for row in range(len(tfidf_array)):
                 # every TF-IDF-value is added to the words overall TF-IDF-count
                self.tfidf_means[col][1] += tfidf_array[row][col]
                if tfidf_array[row][col] != 0.0:
                    totalrows += 1
                    if not mean_mismatch: # include strings without the word?
                        continue
                rowcount += 1
            self.tfidf_means[col][1] /= rowcount # divide by rowcount to get the mean
            self.tfidf_means[col][2] = totalrows # the number of opuses, this work is mentioned for is also stored

        # converting the results to a DataFrame
        self.tfidf_means_df = pd.DataFrame(self.tfidf_means,columns=["Word","TF-IDF","Number of opuses"])
        return self.tfidf_means_df

class AnalysisProcessing():
    """Processes the analysis done before"""

    def __init__(self,info,portfolio,tfidf):
        self.inf = info
        self.pf = portfolio
        self.tfidf = tfidf

    def summary(self, total_words = 10):
        """This function provides a summary of a Text-Analysis done before. Parameters:
        - total_words: represents the total number of words to be listed"""

        # Print an introductionary sentence
        print(f"{self.inf['Name']} (IMDB-ID: {self.inf['imdbID']}) is associated with a total number of {self.inf['Total']} opuses. This person's {(len(self.pf) if len(self.pf) < 10 else 10)} most popular opuses on IMDB are:\n")

         # iterating through the opuses for output
        iterstop = (len(self.pf) if len(self.pf) < 10 else 10)
        for i,op in self.pf[:iterstop].iterrows():
            print(f"{op['Title']} (IMDB-Rating: {op['imdbRating']})")

        # introductionary sentence for the tfidf-values
        print(f"\nThe {total_words} words most used to describe the plots of {(f'the {len(self.pf)} most popular opuses' if len(self.pf) < self.inf['Total'] else 'all the opuses')} are:\n")

        # iterating through the word-tfidf-list
        print(self.tfidf.sort_values(by=["TF-IDF"],ascending=False)[:total_words])

        # plotting the top 10 words according to their tfidf score
        self.visualize("bar",self.tfidf.sort_values(by=["TF-IDF"],ascending=False)["TF-IDF"][:10],xticks=self.tfidf.sort_values(by=["TF-IDF"],ascending=False)["Word"][:10],title=f"Top 10 words with highest TF-IDF score used to describe {self.inf['Name']}'s opuses")

        print("\nBelow you can get an impression of the correlation between the number of opuses the words are used in and their importance (measured through mean TF-IDF):")

        # plotting the relation between the words' tfidf-score and the number of opuses they're used in
        self.visualize("strip",self.tfidf[["Number of opuses","TF-IDF"]],title="Relation between TF-IDF-score and number of opuses the words are used in")
    
    def visualize(self,type,data_df,**kwargs):
        """This function uses plots to visualize the analyzed data.
        parameters:
        - type: indicating the type of visualization: strip for stripplot, bar barplot
        - xvar: the independent/horizontal variable
        - yvar: the dependent/vertical variable"""

        #defines a stripplot
        if type == "strip":
            plt.clf()
            plot = sns.stripplot(x=data_df.columns[0],y=data_df.columns[1],data=data_df,jitter=True)
            plot.xlim=(data_df.iloc[:,0].min(),data_df.iloc[:,0].max())
            plot.ylim=(data_df.iloc[:,1].min(),data_df.iloc[:,1].max())
            if "title" in kwargs:
                plt.title = kwargs.get("title")
            plt.show()
        # defines a barplot
        elif type == "bar":
            plt.clf()
            plot = plt.bar(range(1,len(data_df)+1),data_df)
            if "xticks" in kwargs:
                plt.xticks(range(1,len(data_df)+1), kwargs.get("xticks"), rotation=45)
            if "title" in kwargs:
                plt.title = kwargs.get("title")
            plt.show()


class InputHandler():
    """this class provides a rudimentary method to handle user input regarding the person,
    whose portfolio should be scraped"""
    def __init__(self):
        pass

    def testfunctions(self):
        """this function uses user input to extract a persons portfolio from IMDB and analyse the plot provided by OMDB"""

        # retrieval
        retr = PortfolioRetrieval(API)
        retr.get_portfolio(input("Which person's portfolio do you want to extract?"),int(input("If you want to limit the number of opuses to be scraped, put it in here (otherwise type 0)")))

        # analysis
        insight = DataAnalysis(retr.portfolio)
        insight.analyse_text("Plot")

        # processing the analysis
        process=AnalysisProcessing(retr.info,retr.portfolio,insight.tfidf_means_df)
        process.summary()

interface = InputHandler()
interface.testfunctions()