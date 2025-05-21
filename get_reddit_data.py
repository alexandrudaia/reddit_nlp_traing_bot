import praw
import pandas as pd
from datetime import datetime, timedelta

# Reddit API credentials
CLIENT_ID = 'OAqjoN-JkdMJTtkeJdwI4Q'
CLIENT_SECRET = 'SlONsjlI5s7qjC3JOgAet3zQYrfMJQ'
USER_AGENT = 'script:hourly_discussion_scraper:v1.0'

# Initialize Reddit API connection
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

def scrape_hourly_discussions(subreddits, hours=2000):
    """
    Scrape hourly discussions from specified subreddits for the last `hours` hours.
    
    Args:
        subreddits (list): List of subreddit names to scrape.
        hours (int): Number of hours to look back.
    
    Returns:
        pd.DataFrame: DataFrame containing scraped data.
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    data = []

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        print(f"Scraping subreddit: {subreddit_name}")
        
        # Fetch posts sorted by 'new'
        for submission in subreddit.new(limit=None):  # Fetch posts sorted by 'new'
            post_time = datetime.utcfromtimestamp(submission.created_utc)
            
            # Stop if the post is older than the start time
            if post_time < start_time:
                break
            
            # Collect data for each post
            data.append({
                "date": post_time.date(),
                "hour": post_time.hour,
                "text": f"{submission.title}\n{submission.selftext}",
                "subreddit": subreddit_name
            })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def save_to_excel(df, filename="reddit_hourly_discussions.xlsx"):
    """
    Save the DataFrame to an Excel file.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Name of the Excel file.
    """
    df.to_excel(filename, index=False)
    print(f"Data saved to {filename}")

# Main script
if __name__ == "__main__":
    # Subreddits to scrape
    subreddits = ["CryptoCurrency", "Bitcoin"]
    
    # Scrape data
    discussions_df = scrape_hourly_discussions(subreddits, hours=2000)
    
    # Check if any data was scraped
    if discussions_df.empty:
        print("No discussions found in the specified time frame.")
    else:
        # Save to Excel
        save_to_excel(discussions_df)
