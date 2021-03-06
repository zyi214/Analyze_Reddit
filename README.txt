This web application crawls Subreddit groups and analyzes posts in these groups based on the following user inputs:
- Subreddit group names (currently down)
- Start date and end date for the posts (currently down)
- # of posts to include in each subreddit group (currently down)
- (optional, if user analysis desired) the posts per week threshold for a user to be considered a frequent user

This web application can then produce the analyses below (as selected by the user):
- Statistical correlation among the groups
- The "hot words" (high frequency words) for the groups
- User analysis, including the sentiment of the group and the frequent users

Packages to install beforehand - commands to run:
pip3 install pandas
pip3 install psaw
pip3 install matplotlib
pip3 install vaderSentiment
pip3 install wordcloud
pip3 install scipy
pip3 install seaborn

To run the web app on your local host:
python3 ui/manage.py runserver

** Note that due to API dependency issues, currently please run with recommended subreddits "trump" and "JoeBiden" and select a date range in 2022. **
