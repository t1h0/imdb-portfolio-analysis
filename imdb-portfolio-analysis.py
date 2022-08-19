import requests
import matplotlib.pyplot as plt
import pandas
import re
from lxml import etree
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer as tfidfVect
from sklearn.feature_extraction import text as sklearnText
import typing

# NEEDS TO BE PROVIDED!
OMDB_API = ''


class PortfolioRetrieval():
    """Retrieves the portfolio of a person on IMDB
    """

    def __init__(self, omdb_api: str):
        """Create a PortfolioRetrieval instance to retrieve IMDB portfolios.

        Args:
            omdb_api (str): OMDB API key.
        """
        self.omdb_api = omdb_api
        self.base_url_omdb = f"http://www.omdbapi.com/?apikey={self.omdb_api}&plot=short&"
        self.base_url_imdb = "https://www.imdb.com"

    def get_portfolio(self, person: str, limit: int = 0) -> pandas.DataFrame:
        """Extracts opuses (with IMDB ids, ratings and plot summaries) of a person on IMDB
        by using IMDB and OMDB (to complement with additional data).


        Args:
            person (str): the name of a person on IMDB.
            limit (int, optional): Limits the number of opuses (sorted by popularity) that should be extracted.
            Use this, if you have an OMDB API limit. Defaults to 0 (no limit).

        Returns:
            pandas.DataFrame: DataFrame with opuses as rows and imdbID, title, type (standalone/episode), year, plot and imdb rating as columns.
        """
        # -------------------------------------
        # I. preparation
        # -------------------------------------
        # format the name to be used as a query in IMDB's search
        query = person.strip().replace(" ", "+")

        # -------------------------------------
        # II. extract IMDB id
        # -------------------------------------
        imdbID = self.__extract_pers_id(query)

        # -------------------------------------
        # III. extract portfolio and complete with OMDB info
        # -------------------------------------
        portfolio = self.__complete_portfolio(
            self.__extract_portfolio(imdbID, limit))

        # -------------------------------------
        # IV. add meta info to DataFrame
        # -------------------------------------
        portfolio.name = person.title()  # Naming it after the person
        portfolio.ID = imdbID

        # -------------------------------------
        # V. return
        # -------------------------------------
        return portfolio

    def __extract_pers_id(self, query: str) -> str:
        """Extracts the IMDB id of a person by taking it from the first result in IMDB search.

        Args:
            query (str): to use as query for IMDB name search.

        Returns:
            str: the IMDB id.
        """

        # -------------------------------------
        # I. get (requests.get) the search result on imdb as html (.text)
        # -------------------------------------
        imdb_search_html = requests.get(
            f"{self.base_url_imdb}/find?s=nm&q={query}").text

        # -------------------------------------
        # II. find a-tag which directly descends from a tag with "result_text" class
        #               (IMDB's result-list is formatted as ..<td class="result_text"><a href=..)
        # II. take the link from that extraction
        # -------------------------------------
        imdb_search_href = etree.HTML(imdb_search_html, None).xpath(
            "//td[@class = 'result_text']/a")[0].get("href")

        # -------------------------------------
        # III. extract the persons id by stripping the link
        # -------------------------------------
        return re.sub("/|(name)", "", imdb_search_href)

    def __extract_portfolio(self, imdb_id: str, limit=0) -> pandas.DataFrame:
        """Extracts the movie/tv portfolio of a person, i.e. the name and IMDB id of every opus the person is associated with.
        and returns a panda DataFrame with titles as rows and the IMDB-ID in a column.

        Args:
            imdb_id (str): The IMDB id of the person to extract the portfolio from.
            limit (int, optional): The maximum number of opuses to extract. Defaults to 0 (no limit).

        Returns:
            pandas.DataFrame: A pandas DataFrame with IMDB id, title, type, year, plot description and IMDB rating of the extracted opuses.
        """
        # -------------------------------------
        # I. prepare the base url for the portfolio
        # -------------------------------------
        imdb_portfolio_url = f"{self.base_url_imdb}/search/title/?adult=include&view=simple&role={imdb_id}&"

        # -------------------------------------
        # II. get the total number of opuses by this person
        # -------------------------------------

        # the total number is displayed on every portfolio on IMDB
        # the reduce loading times, it is taken from an overview with only 1 opus displayed ("count=1")
        total_op_html = requests.get(f"{imdb_portfolio_url}count=1").text
        # look for the tag which includes the total number of opuses
        total_op_text = etree.HTML(total_op_html, None).xpath(
            "//span[contains(text(),'1-1 of')]")[0].text
        # format the string to only hold the total number and convert to int
        total_op = int(total_op_text.replace(
            "1-1 of ", "").replace(" titles.", ""))

        # -------------------------------------
        # III. iterate through the pages of the person's portfolio
        # -------------------------------------

        result = []  # results will go in here and be converted into a DataFrame at the end

        # -------------------------------------
        # prepare some scraping limits either
        # - because a manual scraping limit is given, or
        # - to reduce loading times and runtime
        # - because auf IMDB's given limits (max count = 250)
        # -------------------------------------

        if limit == 0 or limit > total_op:
            # set limit to total number of opuses if limit greater than that total or not given
            limit = total_op

        # limit for opuses displayed at a single portfolio site on IMDB (max is 250)
        count = limit if limit < 250 else 250

        # "start" defines the index number of the list's first (top) opus to be displayed in IMDB.
        # The limit is either a given scarping limit or the total amount of opuses by this person
        for start in range(1, limit+1, 250):

            # -------------------------------------
            # get the html
            # the "count"-parameter in IMDB's URL defines, how many opuses should be displayed on one site (max.250)
            # if a scraping-limit is given, this is used for the amount of opuses displayed (to reduce loading time)
            # -------------------------------------
            imdb_movies_html = requests.get(
                f"{imdb_portfolio_url}count={count}&start={start}").text

            # -------------------------------------
            # find every title's IMDB-Link
            # --> criteria: find every a-tag which descends
            #               from a tag with "lister-item-header" class and is the episode's tag in case of a series,
            #               otherwise the movie's tag
            #               (IMDB's portfolio site is formatted as ..<[tag] class="lister-item-header">..<a href=..)
            # -------------------------------------
            imdb_movies_tags = etree.HTML(imdb_movies_html, None).xpath(
                "//span[@class = 'lister-item-header']//a[not(following-sibling::small[text() = 'Episode:']) or preceding-sibling::small[text() = 'Episode:']]")

            # -------------------------------------
            # extract the IMDB ids (inside of the hrefs), the opuses' titles (inside of the tags)
            # and the type of opus from the tags
            # -------------------------------------
            type_xpath = etree.XPath(
                "boolean(./preceding-sibling::small[text() = 'Episode:'])")
            # itereate trough the tags
            for tag in imdb_movies_tags:
                # -------------------------------------
                # define the type of opus:
                # "standalone" for autonomous opuses like movies, Tv specials etc
                # "episode" for an episode inside a series
                # -------------------------------------
                opus_type = ("Episode" if type_xpath(tag) else "Standalone")
                # -------------------------------------
                # append the current tag to the results in a list, by:
                # 1. inserting the imdb-id as first item (by stripping the href, so that the id remains)
                # 2. inserting the title (i.e. the tag's content) as second item
                # 3. inserting the opuses' type as third item
                # 4. inserting three more NaN-items, representing Year,Plot and IMDB-rating of the opus,
                #    to be inserted via OMDB
                # -------------------------------------
                result.append([re.sub("/|(title)", "", tag.get("href")),
                              tag.text, opus_type, "NaN", "NaN", "NaN"])
        # return the results as a pandas DataFrame
        output = pandas.DataFrame(
            result, columns=["imdbID", "Title", "Type", "Year", "Plot", "imdbRating"])
        return output

    def __complete_portfolio(self, portfolio: pandas.DataFrame) -> pandas.DataFrame:
        """Completes a portfolio with Year, Plot and IMDB rating of every title via OMDB API.

        Args:
            portfolio (pandas.DataFrame): A portfolio as returned by self.__extract_portfolio().

        Returns:
            pandas.DataFrame: The completed portfolio.
        """

        portfolio = portfolio.copy()  # work-in-progress portfolio

        # iterate through the rows (=opuses) of the searched person in the DataFrame
        for i, row in portfolio.iterrows():
            # for every opus get the corresponding json from omdb by IMDB id
            omdb_entry = requests.get(
                f"{self.base_url_omdb}i={row['imdbID']}").json()
            # update the last three rows "Year", "Plot" and "imdbRating" with the info included in the omdb json
            for column in portfolio.columns[3:]:
                try:
                    portfolio.at[i, column] = omdb_entry[column]
                except KeyError:
                    pass

        return portfolio


class DataAnalysis():
    """Performs TF-IDF analysis on a corpus stored in a pandas DataFrame column."""

    def __init__(self):
        pass

    def analyze_corpus(self, df: pandas.DataFrame, text_column: str, mean_mismatch: bool = False, ignore: list[str] = []) -> pandas.DataFrame:
        """Takes a DataFrame and analyzes a corpus column containing documents (in this case: plot descriptions)
        by extracting keywords using the "term frequency times inverse document frequency" (TF-IDF) measurement.
        The means of the TF-IDFs are then calculated over the documents and returned in a new DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame containing the corpus in a column.
            text_column (str): The name of the column containing the corpus.
            mean_mismatch (bool, optional): If false, mean calculations are only based on number of documents that actually contain the respective term. Defaults to False.
            ignore (list[str], optional): Can contain a list of terms to be ignored in the analysis process. English stop-words (included in scikit-learn)
                                          and "nan" (e.g. if a plot or the film itself wasn't found in OMDB's database) are automatically added. Defaults to [].

        Returns:
            pandas.DataFrame: Data frame containing per term the respective term, tf-idf mean and number of documents the term appeared in.
        """

        # -------------------------------------
        # I. Generating TF-IDFs
        # -------------------------------------

        # prepare list of terms to be ignored
        stop_words_list = ["nan"]
        if isinstance(ignore, list):
            stop_words_list.extend(ignore)
        # initialize the tfidf-vectorizer (see scikit-learn doc for further details)
        tfidf_vect = tfidfVect(
            stop_words=sklearnText.ENGLISH_STOP_WORDS.union(stop_words_list))
        # generating the tfidfs
        tfidf = tfidf_vect.fit_transform(df.loc[:, text_column])
        # converting the tfidf-result to a usable array
        tfidf_array = tfidf.toarray()
        # this list contains all the terms, with their indices matching the tfidf-list's indices
        tfidf_terms = tfidf_vect.get_feature_names_out()

        # -------------------------------------
        # II. Calculating the Means
        # -------------------------------------

        # this will contain the resulting list, containing lists with terms and their means
        tfidf_means = []

        # iterate through the tfidf-list (columns=terms,rows=documents (i.e. plots))
        for col in range(len(tfidf_array[0])):
            # adding a list to the result list with the term as string and 0 as TF-IDF-value
            tfidf_means.append([tfidf_terms[col], 0, 0])
            rowcount = 0
            totalrows = 0
            for row in range(len(tfidf_array)):
                # every TF-IDF-value is added to the terms overall TF-IDF-count
                tfidf_means[col][1] += tfidf_array[row][col]
                if tfidf_array[row][col] != 0.0:
                    totalrows += 1
                elif not mean_mismatch:  # include documents without the term in mean calculation?
                    continue
                rowcount += 1
            # divide by rowcount to get the mean
            tfidf_means[col][1] /= rowcount
            # the number of documents this term is mentioned in is also stored
            tfidf_means[col][2] = totalrows

        # converting the results to a DataFrame
        return pandas.DataFrame(tfidf_means, columns=["Term", "TF-IDF mean", "Number of documents"])


class AnalysisProcessing():
    """Processes IMDB portfolio analysis done by an object of class DataAnalysis."""

    def __init__(self):
        pass

    def process(self, portfolio: pandas.DataFrame, analysis: pandas.DataFrame, number_of_terms: int = 10):
        """Processes and presents the analysis of a PortfolioRetrieval portfolio done by DataAnalysis.

        Args:
            portfolio (pandas.DataFrame): Portfolio output by PortfolioRetrieval.
            analysis (pandas.DataFrame): Analysis output by DataAnalysis.
            number_of_terms (int, optional): Number of (top) terms to be listed. Defaults to 10.
        """

        name = portfolio.name
        id = portfolio.ID
        total = len(portfolio.index)
        portfolio = portfolio.copy()
        analysis = analysis.copy()

        # Print an introductionary sentence
        print(f"{name} (IMDB-ID: {id}) is associated with a total number of {total} opuses. This person's {(len(portfolio) if len(portfolio) < 10 else 10)} most popular opuses on IMDB are:\n")

        # iterating through the opuses for output
        iterstop = (len(portfolio) if len(portfolio) < 10 else 10)
        for i, op in portfolio[:iterstop].iterrows():
            print(f"{op['Title']} (IMDB-Rating: {op['imdbRating']})")

        # introductionary sentence for the tfidf-values
        print(f"\nThe {number_of_terms} terms most used to describe the plots of {name}'s {(f'most popular ' if len(portfolio) < total else '')}opuses are:\n")

        # iterating through the analysis
        print(analysis.sort_values(
            by=["TF-IDF mean"], ascending=False)[:number_of_terms])

        # plotting the top number_of_terms terms according to their tfidf score
        self.__visualizeCorrelation(analysis.sort_values(by=["TF-IDF mean"], ascending=False)[["Term", "TF-IDF mean"]][:number_of_terms], "strip", xticks=analysis.sort_values(
            by=["TF-IDF mean"], ascending=False)["Term"][:number_of_terms], title=f"Top {number_of_terms} terms with highest TF-IDF score used to describe {name}'s opuses")

        print("\nHere you can get an impression of the correlation between the number of opuses the terms are used in and their importance (measured through mean TF-IDF):")

        # plotting the relation between the terms' tfidf-score and the number of opuses they're used in
        self.__visualizeCorrelation(analysis[["Number of documents", "TF-IDF mean"]], "strip",
                                    title="Relation between TF-IDF-score and number of opuses the terms are used in")

    def __visualizeCorrelation(self, data: pandas.DataFrame, plotType: str, **kwargs):
        """Plots correlation between two variables.

        Args:
            data (pandas.DataFrame): A data frame consisting of two columns, i.e. two variables to display the correlation of.
            plotType (str): The type of visualization: 'strip' for stripplot, 'bar' for barplot.
            **kwargs: Additional arguments to pass to the plot function. Supported are title and xticks (only for barplot).
        """

        # stripplot
        if plotType == "strip":
            plt.clf()
            plot = sns.stripplot(
                x=data.iloc[:, 0], y=data.iloc[:, 1], data=data, jitter=True)
            plot.set_xlim = (data.iloc[:, 0].min(), data.iloc[:, 0].max())
            plot.set_ylim = (data.iloc[:, 1].min(), data.iloc[:, 1].max())
            if "title" in kwargs:
                plt.title = kwargs.get("title")
            plt.show()

        #  barplot
        elif plotType == "bar":
            plt.clf()
            plot = plt.bar(range(1, len(data)+1), data)
            if "xticks" in kwargs:
                plt.xticks(range(1, len(data)+1),
                           kwargs.get("xticks"), rotation=45)
            if "title" in kwargs:
                plt.title = kwargs.get("title")
            plt.show()


class InputHandler():
    """This class provides a basic method to handle user search input."""

    def __init__(self):
        pass

    def testFunctions(self):
        """This function uses user input to extract a person's portfolio from IMDB and to analyze the plot provided by OMDB."""

        # retrieval
        retr = PortfolioRetrieval(OMDB_API)
        portfolio = retr.get_portfolio(input("Which person's portfolio do you want to extract? "), int(
            input("If you want to limit the number of opuses to be scraped, put it in here (otherwise type 0) ")))

        # analysis
        insight = DataAnalysis()
        analysis = insight.analyze_corpus(portfolio, "Plot")

        # processing the analysis
        processing = AnalysisProcessing()
        processing.process(portfolio, analysis)


interface = InputHandler()
interface.testFunctions()
