import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

csv_file_path = 'C:/Users/User/Documents/Dissertation Research/Literature review/Literature Review for CODING.csv'
output_filename = 'heatmap_unique_models_per_study_area.png'
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

# count a model only once per area within each study, we define a unique
# entry by the combination of these three columns.
df_unique = df.drop_duplicates(subset=['Study (#)', 'Area', 'Models used'])
print("Data processed to count unique models per study and area.")

# Heatmap Data 
# cross-tabulation (contingency table) from this new, unique data
heatmap_data = pd.crosstab(df_unique['Area'], df_unique['Models used'])

print("Generating Heatmap")
try:
    # Set up the matplotlib figure for better visualization
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5, linecolor='black')
  
    plt.title('Heatmap: Unique Models Used per Research Area (per Study)', fontsize=16)
    plt.xlabel('Models Used', fontsize=12)
    plt.ylabel('Research Area', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    full_save_path = os.path.join(save_folder_path, output_filename)
    plt.savefig(full_save_path, dpi=300)
    plt.show()
    plt.close()

    print(f"\nHeatmap successfully saved to Downloads folder:")
    print(full_save_path)

except Exception as e:
    print(f"\nAn error occurred while generating or saving the plot: {e}")

