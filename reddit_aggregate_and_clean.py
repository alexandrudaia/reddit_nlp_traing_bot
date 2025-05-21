import pandas as pd
import re
from datetime import datetime

def clean_text(text):
    """Clean and standardize text content."""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove Reddit-style links
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Remove multiple punctuation
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    return text.strip()

def aggregate_hourly_data(input_file, output_file):
    """
    Aggregate Reddit discussion data by hour and clean the text content.
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path to output CSV file
    """
    try:
        # Read the Excel file
        print("Reading input file...")
        df = pd.read_excel(input_file)
        print(f"Initial shape: {df.shape}")
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Create a datetime column combining date and hour
        df['datetime'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
        
        # Clean the text column
        print("Cleaning text content...")
        df['text'] = df['text'].apply(clean_text)
        
        # Group by datetime and subreddit, aggregate text
        print("Aggregating data by hour...")
        df_aggregated = df.groupby(['datetime', 'subreddit']).agg({
            'text': lambda x: ' || '.join(x[x != '']),  # Join non-empty texts with separator
        }).reset_index()
        
        # Sort by datetime
        df_aggregated = df_aggregated.sort_values('datetime')
        
        # Extract date and hour from datetime for final output
        df_aggregated['date'] = df_aggregated['datetime'].dt.date
        df_aggregated['hour'] = df_aggregated['datetime'].dt.hour
        
        # Reorder columns and drop datetime
        final_df = df_aggregated[['date', 'hour', 'subreddit', 'text']]
        
        # Save to CSV
        print(f"Saving aggregated data to {output_file}...")
        final_df.to_csv(output_file, index=False)
        
        print("\nAggregation Summary:")
        print(f"Original number of rows: {len(df)}")
        print(f"Aggregated number of rows: {len(final_df)}")
        print(f"Number of unique hours: {len(final_df['hour'].unique())}")
        print(f"Number of unique dates: {len(final_df['date'].unique())}")
        print("\nFirst few rows of aggregated data:")
        print(final_df.head())
        
        return True
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "reddit_hourly_discussions.xlsx"
    output_file = "aggregated_reddit_discussions.csv"
    
    success = aggregate_hourly_data(input_file, output_file)
    
    if success:
        print("\nData aggregation completed successfully!")
    else:
        print("\nData aggregation failed. Please check the error message above.")
