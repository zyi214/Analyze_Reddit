#https://github.com/pushshift/api
#https://github.com/dmarx/psaw
from psaw import PushshiftAPI
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import os
from wordcloud import WordCloud, STOPWORDS
from find_hotwords import find_hot_words, BUNDLE_REDDITS
from Stats import find_relations

MY_PATH = os.path.abspath(__file__)
MY_PATH = MY_PATH.replace('analyze_reddit.py', '')
ANALYZER = SentimentIntensityAnalyzer()
api = PushshiftAPI()
BOT_WORDS = ['@', 'LIVE']
plt.switch_backend('Agg') 



def calculate_sentiment(text):
    '''
    Calculates sentiment score of a string.
    Inputs:
        text(str): the text to be analyzed
    Returns:
        an int between 1(very positive) and -1(vert negative)
    '''
    return ANALYZER.polarity_scores(text)['compound']


def test_if_repost(group_url, title, text, url):
    '''
    Test of the post is a repost of some kind.
    Inputs:
        group_url(string): id of the subreddit
        title(string): title of post
        text(string): body of the post
        url(url): url of the post
    '''
    if (group_url not in url) and \
        ((text == '') or (text == '[removed]') or \
        (any(word in title for word in BOT_WORDS))):
        return True
    return False


class Subreddit:

    def __init__(self, name, start_date, end_date, n):
        '''
        Construct a new Reddit_Groups object with information about the
        groups' posts and users.

        Groupnames(list): list of reddit groups to analyze
        '''
        self.name = name
        self.posts = self.create_df(start_date, end_date, n)
        #the following variables are created from the actual data we get because
        #there are some mysterious limits to the api
        self.min_epoch = self.posts.epoch_time.min()
        self.max_epoch = self.posts.epoch_time.max()
        self.n = len(self.posts)


    def get_posts(self, start_date, end_date, n):
        '''
        Extract posts from a subreddit with criteria.
        Inputs:
            start_date (inclusive), end_date (exclusive)(lists): two lists of
              three elements contaiining year, month, date of limit to extract
              the posts. Exp. [2020, 1, 1].
            n (int): limit on number of posts to extract
        Returns: a list of submissions
        '''
        year1, month1, day1 = start_date
        start_epoch = int(dt.datetime(year1, month1, day1).timestamp())
        year2, month2, day2 = end_date
        end_epoch = int(dt.datetime(year2, month2, day2).timestamp())
        return list(api.search_submissions(before = end_epoch, after = start_epoch, \
            subreddit = self.name, filter=['id', 'title', 'author', 'selftext', \
            'score', 'url'], limit = n))


    def create_df(self, start_date, end_date, n):
        '''
        Create a pandas dataframe from a list of submissions.
        Inputs:
            start_date (inclusive), end_date (exclusive)(lists): two lists of
              three elements contaiining year, month, date of limit to extract
              the posts. Exp. [2020, 1, 1].
            n (int): limit on number of posts to extract
        Returns: a pandas dataframe with author, post title, post content,
          time the post was created (ymd, hour, minute, second) and sentiment
          score of the post
        '''
        submissions = self.get_posts(start_date, end_date, n)
        l_id, l_title, l_author, l_selftext, l_score, l_url, l_time, l_sent,\
        = [], [], [], [], [], [], [], []
        for sub in submissions:
            l_id.append(sub.id)
            l_title.append(sub.title)
            l_author.append(sub.author)
            l_score.append(sub.score)
            l_time.append(sub.created_utc)
            if hasattr(sub, 'selftext'):
                l_selftext.append(sub.selftext)
                l_sent.append(calculate_sentiment(sub.title + sub.selftext))
            else:
                l_selftext.append('')
                l_sent.append(calculate_sentiment(sub.title))
            if hasattr(sub, 'url'):
                l_url.append(sub.url)
            else:
                l_url.append[None]

        df = pd.DataFrame({'id': l_id, 'title': l_title, 'author': l_author, \
            'text': l_selftext,'score': l_score, 'url': l_url, 'epoch_time':l_time, \
            'sentiment_score': l_sent})
        df['time'] = df.apply(lambda row: dt.datetime.fromtimestamp(row['epoch_time']), axis = 1)
        ymd, hour, minute, second = \
        zip(*[(d.strftime("%Y%m%d"), d.hour, d.minute, d.second) for d in df['time']])
        df = df.assign(ymd = [int(i) for i in ymd], hour = hour, \
            minute = minute, second = second)
        return df


def get_group_data(names, start_date, end_date, n, plot_title):
    '''
    Compare sentiment scores vs. time for any number of subreddits.
    Inputs:
        names(list): names of the subreddit groups
        start_date, end_date (list): two lists of three elements contaiining
          year, month, date of limit to extract the posts
        n (int): limit on number of posts to extract
        plot_title(str): name of plot, also filename of .png file
    Returns:
        a list of pandas DataFrame of all subreddits, with author, post title,
          post content, time the post was created (ymd, hour, minute, second)
          and sentiment score of the post.
    Outputs:
        plot_title.png file with scores vs. time for all subreddits
    Sample use:  k = analyze_reddit.get_group_data(['SandersForPresident', '
    Pete_Buttigieg'], [2020, 2,6], [2020, 2, 12], 50000, 'bernie_vs_pete')
    '''
    min_epoch = 0
    max_epoch = 10**11
    subreddits = []
    plt.clf()
    ax = plt.axes()
    min_sent = 1
    max_sent = -1
    post_count_groups = []
    for name in names:
        sub = Subreddit(name, start_date, end_date, n)
        subreddits.append(sub)
        if sub.min_epoch > min_epoch:
            min_epoch = sub.min_epoch
        if sub.max_epoch < max_epoch:
            max_epoch = sub.max_epoch
        df = sub.posts.copy()
        df = df[(df['epoch_time'] >= min_epoch) & (df['epoch_time'] <= max_epoch)]
        df_group = df.groupby('ymd', as_index=False)
        post_count_groups.append(list(df_group.size()))
        mean_score = df_group['sentiment_score'].mean()
        plt.plot(mean_score.ymd, mean_score.sentiment_score, '-o', label = sub.name)
        min_sent = min(min_sent, mean_score.sentiment_score.min())
        max_sent = max(max_sent, mean_score.sentiment_score.max())
    sent_range = max_sent - min_sent
    l_change_all = []
    for sub in subreddits:
        l_change = []
        dates = get_changes(sub)
        if len(dates) != 0:
            for date, sent, diff in dates:
                if diff < 0:
                    ax.arrow(date, sent - sent_range/5, 0, sent_range/10, \
                        head_width=0.1, head_length=sent_range/40)
                if diff > 0:
                    ax.arrow(date, sent + sent_range/5, 0, -sent_range/10, \
                        head_width=0.1, head_length=sent_range/40)
                l_change.append(date)
        l_change_all.append(l_change)
    ax.ticklabel_format(useOffset=False, style = 'plain')
    plt.xticks(rotation=90)
    plt.legend(loc='best')
    plt.savefig(MY_PATH + 'mysite/static/graphs/' + plot_title + '.png',\
     bbox_inches='tight', dpi = 300)
    return (post_count_groups, l_change_all, subreddits)


def get_users_data(subreddits, post_per_week = 10):
    '''
    Analyze users who post frequently in all groups.
    inputs:
        subreddits(list): a list of Subreddit objects
        post_per_week(int): minimum number of posts per week to consider a user
          a 'frequent poster'
    '''
    headers = ['Author', 'Is_bot', 'Mean_sentiment_score', 'Sentiment_score_std', \
    'Number_of_posts']
    freq_posters = []
    for sub in subreddits:
        wk = (sub.max_epoch - sub.min_epoch)/604800
        n_post = max(1, post_per_week * wk)
        df = sub.posts.copy()
        grouped = df.groupby('author')
        pcount = grouped['sentiment_score'].agg([np.mean, np.std, 'count'])
        pcount = pcount[pcount['count'] > n_post]
        if len(pcount) == 0:#
            return None#
        pcount['author'] = list(pcount.index)
        pcount['is_bot'] = pcount.apply(lambda row: is_bot(sub, row.author), \
            axis = 1)
        pcount = pcount[['author', 'is_bot', 'mean', 'std', 'count']]
        pcount = pcount.sort_values(by='count')
        pcount = pcount.round(4)
        freq_posters.append([l.tolist() for l in pcount.values])
    return (headers, freq_posters)


def is_bot(sub, username):
    '''
    Check if a user is a bot (defined as someone who only reposts news/videos)
    '''
    group_url = "www.reddit.com/r/" + sub.name
    df = sub.posts[sub.posts.author == username].copy()
    df['is_repost'] = df.apply(lambda row: \
        test_if_repost(group_url, row['title'], row['text'], row['url']), axis = 1)
    if len(df[df.is_repost == True])/len(df) >= 0.8:
        return True
    return False


def get_comments(subreddit, linkid):
    '''
    Get comments of the a subreddit post.
    Inputs:
        subreddit(str): name of the group
        linkid: link_id of the post to get comments
    '''
    a = list(api.search_comments(subreddit = subreddit, link_id = linkid, \
        limit = 10000))


def get_popular_posts(subreddit, day):
    '''
    Get popular posts of the day of that subreddit
    '''
    posts = subreddit.posts[subreddit.posts['ymd'] == day]


def get_changes(subreddit):
    '''
    '''
    l_change = []
    sent = subreddit.posts.groupby('ymd')['sentiment_score'].mean()
    i = 1
    while i < len(sent):
        diff = sent.iloc[i - 1] - sent.iloc[i]
        if np.abs(diff) > 0.1:
            l_change.append((sent.index[i], sent.iloc[i], diff))
        i += 1
    return l_change


def get_posts(subreddits, dates):
    i = 0
    all_popular_posts = []
    for sub_dates in dates:
        group_posts = []
        for date in sub_dates:
            posts = subreddits[i].posts
            posts = posts[posts.ymd == date]
            most_popular = posts.sort_values('score', ascending = False).iloc[0]
            group_posts.append(['Popular post of '+ str(date) + 'is: "'+
                most_popular.title + '",upvotes:'+ most_popular.score])
        i += 1
        all_popular_posts.append(group_posts)
    return all_popular_posts



def get_correlations(subreddits):
    '''
    Get correlation of subbreddit groups
    '''
    dfs = []
    sub_names = []
    start_date = dt.datetime.fromtimestamp(subreddits[0].min_epoch)
    start_date = (start_date.year, start_date.month,start_date.day)
    end_date = dt.datetime.fromtimestamp(subreddits[0].max_epoch)
    end_date = (end_date.year, end_date.month, end_date.day)
    for sub in subreddits:
        dfs.append(sub.posts)
        sub_names.append(sub.name)
    sig_score = 0.5
    return find_relations(dfs, sub_names, sig_score, start_date, end_date)


def create_word_cloud(subreddits, k):
    '''
    Generate word clouds of subresddit groups
    '''
    l_picnames = []
    words = find_hot_words(subreddits, k)
    for key, item in words.items():
        wordcloud = WordCloud(stopwords = STOPWORDS, width = 1000, \
            height = 500, background_color="white").\
        generate_from_frequencies(dict(item))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud)
        fig1 = plt.gcf()
        l_picnames.append(key + '_wordcloud.png')
        fig1.savefig(MY_PATH + 'mysite/static/graphs/' + key + '_wordcloud.png', \
            dpi = 100, bbox_inches='tight')
    return l_picnames


def go(names, start_date, end_date, n, post_per_week, plot_title, \
    analyze_user=False, analyze_words=False, analyze_correlation=False):
    '''
    GO!
    k = analyze_reddit.go(['SandersForPresident', 'Pete_Buttigieg'],
    [2020, 2,6], [2020, 2, 12], 50000, 10, 'bernie_vs_pete', True)
    '''
    results = []
    post_count_groups, l_change_all, subreddits = \
    get_group_data(names, start_date, end_date, n, plot_title)
    if analyze_user == False:
        results.append(None)
    else:
        results.append(get_users_data(subreddits, post_per_week = 10))
    if analyze_words == False:
        results.append(None)
    else:
        results.append(create_word_cloud(subreddits, 50))
    if analyze_correlation == False:
        results.append(None)
    else:
        results.append(post_count_groups + get_posts(subreddits, l_change_all) +\
         get_correlations(subreddits))
    return results
