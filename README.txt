This web application crawls Subreddit groups and analyzes posts in these groups based on the following user inputs:
- Subreddit group names
- Start date and end date for the posts
- # of posts to include in each subreddit group
- (optional, if frequent user analysis desired) the posts per week threshold for a user to be considered a frequent user

This web application can then produce the analyses below (as selected by the user):
- Statistical correlation among the groups
- The "hot words" (high frequency words) for the groups
- Frequent users of the groups

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
