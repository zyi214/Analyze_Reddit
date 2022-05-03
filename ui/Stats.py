# statistical correlation scores 

#https://github.com/pushshift/api
#https://github.com/dmarx/psaw

#https://campus.datacamp.com/courses/visualizing-time-series-data\
#-in-python/work-with-multiple-time-series?ex=9

import datetime as dt
import pandas as pd
from datetime import date, timedelta

from scipy.stats.stats import pearsonr
import seaborn as sns 

 

def df_time(dfs, start_date, end_date):
    """
    Filter the dataframes called from api with time limit again, to decrease the number of 
    calls to api, decreasing running time

    Inputs: 
        dfs: list of dataframes
        start_date, end_date (list): two lists of three elements containing
            year, month, date of limit to extract the posts
    Outputs:
        dfs: list of dataframes filtered through time limits
    """
    for df in dfs:
        df.ymd = pd.to_numeric(df.ymd)
        index_names_s = df[df["ymd"] < start_date].index
        index_names_l = df[df["ymd"] > end_date].index
        df.drop(index_names_s, inplace = True)
        df.drop(index_names_l, inplace = True)
    return dfs

def get_correlation(dfs, subreddits, sig_score):
    '''
    Compare sentiment scores vs. time for any number of subreddits.
    Inputs:
        dfs(list of dataframes): a list of dataframes returned by 
            compare_group_sent() function
        subreddits(list): names of the subreddit groups
        sig_score(flt): a number in the range of [0,1], that filters out the 
            correlation scores that are considered insignificant
    Outputs:
        a list of tuples reflecting the correlation scores between any two groups 
        that are considered significant (filtered by the sig_score input), under the 
        form of (group, group, correlation score)
    Sample use:  
        s = Stats.get_correlation(analyze_reddit.compare_group_sent(['SandersForPresident', '\
        Pete_Buttigieg'], [2020, 2,6], [2020, 2, 12], 50000, "bernie_vs_pete"), ['Sa\
        ndersForPresident', 'Pete_Buttigieg'], 0.1) 
    '''
    # construct a pandas dataframe with the the mean of sentiment score mean per day, 
    # with different columns signifying the data for a different subreddit
    
    df_data = pd.DataFrame()
    for i in range(len(dfs)):
        df_count = dfs[i].groupby('ymd', as_index=False)['sentiment_score']\
        .mean().sentiment_score
        df_data[subreddits[i]] = df_count
    corr_matrix = df_data.corr(method = "pearson")
    sentences = []
    output = []
    #iterate over half of the matrix through ith row and jth column
    for i in range(len(corr_matrix)):
        for j in range(i):
            if abs(corr_matrix.iloc[i][j]) >= sig_score:
                tup = (subreddits[i], subreddits[j], corr_matrix.iloc[i][j])
                output.append(tup)
    return output

def tell_correlation(tuples):
    """
    Translate the information from a list of tuples returned by get_correlation into
    readable sentences

    Inputs:
        tuples: a list of tuples
    Outputs:
        A list of sentences
    Sample Use:
        sentences = Stats.tell_correlation(Stats.get_correlation(Stats.compare_group_sent([\
        'SandersForPresident', 'Pete_Buttigieg'], [2020, 2,6], [2020, 2, 12], 50000, \
        "bernie_vs_pete"), ['SandersForPresident', 'Pete_Buttigieg'], 0.1))
    """
    sentences = []
    for tup in tuples:
        say = "The correlation score between group " + str(tup[0]) + \
        " and group " + str(tup[1]) + " is: " + str(tup[2])
        sentences.append(say)
    return sentences

def find_relations(dfs, subreddits, sig_score, start_date, end_date):
    """
    To extract period of time for which the correlation between the two correlation 
    scores are significant (bigeer than or equal to the sig_score), for a period of 
    at least more than 3 days

    Inputs:
        dfs(list of dataframes): a list of dataframes returned by 
            compare_group_sent() function
        subreddits(list): names of the subreddit groups
        sig_score(flt): a number in the range of [0,1], that filters out the 
            correlation scores that are considered insignificant
        start_date, end_date (list): two lists of three elements containing
            year, month, date of limit to extract the posts
    Outputs:
        a list of lists in the format of each element being [group1, group2, correlation score, 
            start_period, end_period]
    Sample Use:
        dd =  Stats.find_relations(analyze_reddit.compare_group_sent(['SandersForPresident', '\
        Pete_Buttigieg'], [2020, 2,6], [2020, 2, 20], 50000, "bernie_vs_pete"), ['Sa\
        ndersForPresident', 'Pete_Buttigieg'], 0.8, (2020,2,6), (2020,2,20)) 

    """

    d0 = dt.date(start_date[0], start_date[1], start_date[2])
    d1 = dt.date(end_date[0], end_date[1], end_date[2])
    delta = d1 - d0

    dic = {}
    from datetime import date, timedelta

    dates = []
    delta_n = timedelta(days=1)
    while d0 <= d1:
        dates.append(d0.strftime("%Y-%m-%d"))
        d0 += delta_n
    
    dates_str = []
    for dd in dates:
        dates_str.append(str(dd[:4]) + str(dd[5:7] + str(dd[8:10])))
    
    for num, sd in enumerate(dates_str[:len(dates_str) - 3]):
        for i in range(3 , delta.days -3):
            dfs_n = []
            for df in dfs:
                date_range = []
                for n in range(num, num+i):
                    if num + i <= len(dates_str):
                        date_range.append(dates_str[n])
                df_n = df.loc[df["ymd"].isin(date_range)]
                dfs_n.append(df_n)

            for tup in get_correlation(dfs_n, subreddits, sig_score):
                dic[tup] = [sd, str(int(sd) + i)]
    mlist = []
    for key in dic.keys():
        mlist.append([key[0], key[1], key[2], dic[key][0], dic[key][1]])

    ind = []
    for i in range(len(mlist)):
        ind.append(i)
        
    bad = []
    for i in ind:
        for j in ind:
            if i != j:
                if (mlist[i][0] == mlist[j][0]) and (mlist[i][1] == mlist[j][1]):
                    if (mlist[i][3] <= mlist[j][3]) and (mlist[i][4] >= mlist[j][4]):
                        bad.append(mlist[j])
    real_bad = []
    for i in bad:
        if i not in real_bad:
            real_bad.append(i)
    for b in real_bad:
        mlist.remove(b)

    return mlist