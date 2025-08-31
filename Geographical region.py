import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

csv_file_path = 'C:/Users/User/Documents/Dissertation Research/Literature review/Literature Review for CODING.csv'
output_filename = 'geographical_region_distribution_study.png'
save_folder_path = os.path.join(os.path.expanduser("~"), "Downloads")

print(f"Loading data from: {csv_file_path}")
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: The file was not found at '{csv_file_path}'. Please check the path.")
    sys.exit()
except Exception as e:
    print(f"An error occurred while loading the CSV file: {e}")
    sys.exit()

#Identify unique (Study, Geographical region) pairs.
#Each study is counted only once for each geographical region.
unique_study_regions = df[['Study (#)', 'Geographical region']].drop_duplicates()

#Count the occurrences of each 'Geographical region' from these unique pairs.
region_counts = unique_study_regions['Geographical region'].value_counts()
print("Data processed to count unique studies per geographical region.")

#Bar Chart
print("\n--- Generating Bar Chart ---")
try:
    #matplotlib figure
    plt.figure(figsize=(10, 7))
    sns.barplot(x=region_counts.index, y=region_counts.values, palette='crest')

    plt.title('Distribution of Studies by Geographical Region', fontsize=16)
    plt.xlabel('Geographical Region', fontsize=12)
    plt.ylabel('Number of Unique Studies', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    full_save_path = os.path.join(save_folder_path, output_filename)
  
    plt.savefig(full_save_path, dpi=300)
    plt.show()
    plt.close()

    print(full_save_path)

except Exception as e:
    print(f"\nAn error occurred while generating or saving the plot: {e}")
