import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

csv_file_path = 'C:/Users/User/Documents/Dissertation Research/Literature review/Literature Review for CODING.csv'
output_filename = 'area_distribution_by_study.png'
save_folder_path = os.path.join(os.path.expanduser("~"), "Downloads")

print(f"--- Loading data from: {csv_file_path} ---")
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: The file was not found at '{csv_file_path}'. Please check the path.")
    sys.exit()
except Exception as e:
    print(f"An error occurred while loading the CSV file: {e}")
    sys.exit()

# Identify unique (Study, Area) pairs.
# each study is counted only once for each research area it covers.
unique_study_areas = df[['Study (#)', 'Area']].drop_duplicates()

#Count the occurrences of each 'Area' from these unique pairs.
area_counts = unique_study_areas['Area'].value_counts()
print("Data processed to count unique studies per research area.")

# Bar Chart
print("\n--- Generating Bar Chart ---")
try:

    plt.figure(figsize=(10, 7))
    sns.barplot(x=area_counts.index, y=area_counts.values, palette='viridis')

    plt.title('Distribution of Research Areas Covered by Unique Studies', fontsize=16)
    plt.xlabel('Research Area', fontsize=12)
    plt.ylabel('Number of Unique Studies', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    full_save_path = os.path.join(save_folder_path, output_filename)
    plt.savefig(full_save_path, dpi=300)
    plt.show()
    plt.close()

    print(f"\nBar chart successfully saved to Downloads folder:")
    print(full_save_path)

except Exception as e:
    print(f"\nAn error occurred while generating or saving the plot: {e}")