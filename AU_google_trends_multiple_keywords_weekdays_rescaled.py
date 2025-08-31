import pandas as pd
from pytrends.request import TrendReq
import time
from datetime import datetime, timedelta

def get_multi_keyword_weekday_trends(keywords_list, start_year, start_mon, start_day, end_year, end_mon, end_day, geo='AU'):
    """
    Fetches, rescales, and filters Google Trends data for a list of keywords, returning a single DataFrame.
    Includes a retry mechanism to handle HTTP 429 errors.
    Args:
        keywords_list (list): A list of keywords to search for.
        start_year (int): The start year of the period.
        start_mon (int): The start month of the period.
        start_day (int): The start day of the period.
        end_year (int): The end year of the period.
        end_mon (int): The end month of the period.
        end_day (int): The end day of the period.
        geo (str): The geography for the trends search 
        Returns:
         A DataFrame containing rescaled weekday trend data for all keywords, or an empty DataFrame.
    """
    pytrends = TrendReq(hl='en-GB', tz=0)
    start_date = datetime(start_year, start_mon, start_day)
    end_date = datetime(end_year, end_mon, end_day)
    full_timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    #HELPER FUNCTION FOR MAKING REQUESTS WITH RETRY LOGIC
    def make_request_with_retry(payload_builder, max_retries=5):
        retries = 0
        while retries < max_retries:
            try:
                payload_builder()
                return pytrends.interest_over_time()
            except Exception as e:
                if '429' in str(e):
                    retries += 1
                    wait_time = 10 * (2 ** retries)
                    print(f"Rate limit hit (429). Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"An unexpected error occurred: {e}")
                    return None
        print(" Max retries exceeded. Failed to fetch data for this request.")
        return None

    # LOOP THROUGH EACH KEYWORD
    all_keywords_final_data = []
    for keyword in keywords_list:
        print(f"\n{'='*50}\nProcessing keyword: '{keyword}'\n{'='*50}")

        # FETCH THE MASTER (WEEKLY) TREND for the current keyword 
        print(f"Fetching master weekly trend for '{keyword}' to use for scaling...")
        master_trend_df = make_request_with_retry(
            lambda: pytrends.build_payload(kw_list=[keyword], cat=0, timeframe=full_timeframe, geo=geo)
        )
        if master_trend_df is None or master_trend_df.empty:
            print(f"Could not fetch master trend for '{keyword}'. Skipping this keyword.")
            continue # Move to the next keyword

        # FETCH DAILY DATA IN CHUNKS for the current keyword 
        all_rescaled_dfs = []
        date_chunks = pd.date_range(start=start_date, end=end_date, freq='3MS').to_pydatetime().tolist()
        if date_chunks[-1] < end_date:
            date_chunks.append(end_date)

        print(f"Fetching and rescaling daily data in chunks for '{keyword}'...")
        for i in range(len(date_chunks) - 1):
            chunk_start = date_chunks[i]
            chunk_end = date_chunks[i+1]
            timeframe = f"{chunk_start.strftime('%Y-%m-%d')} {chunk_end.strftime('%Y-%m-%d')}"
            
            daily_chunk_df = make_request_with_retry(
                lambda: pytrends.build_payload(kw_list=[keyword], cat=0, timeframe=timeframe, geo=geo))

            if daily_chunk_df is not None and not daily_chunk_df.empty:
                if 'isPartial' in daily_chunk_df.columns:
                    daily_chunk_df = daily_chunk_df.drop(columns=['isPartial'])
                
                master_chunk = master_trend_df[(master_trend_df.index >= chunk_start) & (master_trend_df.index < chunk_end + timedelta(days=1))]
                if not master_chunk.empty:
                    scaling_factor = master_chunk[keyword].mean() / 100.0
                    daily_chunk_df[keyword] = daily_chunk_df[keyword] * scaling_factor
                    all_rescaled_dfs.append(daily_chunk_df)
                else:
                    print(f"Could not find master trend data for chunk. Skipping.")
            else:
                print(f"No daily data returned for timeframe {timeframe}.")
            time.sleep(2) # delay between chunk requests

        if not all_rescaled_dfs:
            print(f"No daily data could be fetched for '{keyword}'. Skipping.")
            continue

        # COMBINE AND CLEAN DATA for the current keyword
        keyword_final_df = pd.concat(all_rescaled_dfs)
        keyword_final_df = keyword_final_df[~keyword_final_df.index.duplicated(keep='first')]
        keyword_final_df = keyword_final_df.rename(columns={keyword: f"{keyword}_rescaled"})
        all_keywords_final_data.append(keyword_final_df)
        print(f"--- Successfully processed keyword: '{keyword}' ---")

    # COMBINE ALL KEYWORDS INTO ONE DATAFRAME
    if not all_keywords_final_data:
        print("\nNo data could be fetched for any of the keywords.")
        return pd.DataFrame()

    print("\nCombining data for all keywords...")
    combined_df = pd.concat(all_keywords_final_data, axis=1)

    # FILTER FOR WEEKDAYS
    print("Filtering combined data for weekdays (Mon-Fri)...")
    is_weekday = combined_df.index.dayofweek < 5
    weekdays_df = combined_df[is_weekday].copy() 
    
    print("\n--- All keywords processed and combined! ---")
    return weekdays_df

if __name__ == '__main__':
    keywords_to_search = [
        'BHP Group', 'BHP', 'BHP stock', 'CSL Limited', 'CSL', 
        'CSL stock', 'debt', 'stocks', 'portfolio', 'inflation', 'risk', 
        'dividend', 'journal', 'value', 'finance', 'capital', 'option' ]
    
    all_keywords_data = get_multi_keyword_weekday_trends(
        keywords_list=keywords_to_search,
        start_year=2020, start_mon=1, start_day=1,
        end_year=2025, end_mon=7, end_day=1,
        geo='AU' )

    if not all_keywords_data.empty:
        print(f"\nTotal weekday data points fetched: {len(all_keywords_data)}")
        
        print("\nFirst 5 Rows of Combined Rescaled Data ---")
        print(all_keywords_data.head())
        
        print("\nLast 5 Rows of Combined Rescaled Data ---")
        print(all_keywords_data.tail())

        try:
            csv_filename = "google_trends_multiple_keywords_AU_weekdays_rescaled.csv"
            all_keywords_data.to_csv(csv_filename)
            print(f"\nData successfully saved to '{csv_filename}'")
        except Exception as e:
            print(f"\nCould not save data to CSV: {e}")



