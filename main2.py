import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import pathlib
from openpyxl import Workbook
from openpyxl.styles import Font
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Function to convert an image to black and white
def convert_to_black_and_white(image_path):
    image = Image.open(image_path).convert('RGB')
    np_image = np.array(image)
    black_mask = (np_image[:, :, 0] == 0) & (np_image[:, :, 1] == 0) & (np_image[:, :, 2] == 0)
    bw_image = np.ones_like(np_image) * 255
    bw_image[black_mask] = [0, 0, 0]
    return Image.fromarray(bw_image)

# Function to count black pixels in an image
def count_black_pixels(image):
    np_image = np.array(image.convert('L'))
    return np.sum(np_image == 0)

# Function to compare two images based on black pixel count and SSIM
def compare_images(image_path1, image_path2, tolerance, ssim_threshold):
    bw_image1 = convert_to_black_and_white(image_path1)
    bw_image2 = convert_to_black_and_white(image_path2)
    
    black_pixels_image1 = count_black_pixels(bw_image1)
    black_pixels_image2 = count_black_pixels(bw_image2)
    
    pixel_difference = abs(black_pixels_image1 - black_pixels_image2)
    ssim_index, _ = ssim(np.array(bw_image1.convert('L')), np.array(bw_image2.convert('L')), full=True)

    return ssim_index, pixel_difference, ssim_index > ssim_threshold and pixel_difference > 0 and pixel_difference <= tolerance and ssim_index < 1

# Parameters
tolerance = 400
ssim_threshold = 0.95

# Path to the main data directory
data_dir = pathlib.Path('../Data_test')

# Collect image paths from all subfolders
subfolders = [f for f in data_dir.iterdir() if f.is_dir()]
image_paths_dict = {subfolder: list(subfolder.glob('*.jpg')) + list(subfolder.glob('*.jpeg')) + list(subfolder.glob('*.png')) for subfolder in subfolders}

# Prepare for Excel output
results = []
similarity_data = []

# Function to compare images in a subsequent folders
def compare_images_in_folder(subfolder, image_paths, all_features_dict, tolerance, ssim_threshold, compared_pairs):
    folder_results = []
    
    # Add the starting comparison marker for the current folder
    folder_results.append({'Image 1': f'Starting comparisons for folder: {subfolder.name}', 'Image 2': ''})
    
    subfolder_list = list(all_features_dict.keys())
    current_index = subfolder_list.index(subfolder)
    
    for i in tqdm(range(len(image_paths)), desc=f"Processing {subfolder.name} folder", ncols=100):
        image_1_path = os.path.normpath(str(image_paths[i])).lower()
        
        # Compare with images in subsequent folders
        for j in range(current_index + 1, len(subfolder_list)):
            other_subfolder = subfolder_list[j]
            other_image_paths = all_features_dict[other_subfolder]
            
            for k in range(len(other_image_paths)):
                image_2_path = os.path.normpath(str(other_image_paths[k])).lower()
                
                # Create a sorted tuple of the pair to ensure uniqueness
                pair = tuple(sorted([image_1_path, image_2_path]))
                
                # Check if the pair has already been compared
                if pair not in compared_pairs:
                    # Perform the comparison
                    ssim_index, pixel_difference, are_similar = compare_images(image_1_path, image_2_path, tolerance, ssim_threshold)
                    
                    # Add the pair to the set of compared pairs
                    compared_pairs.add(pair)
                    
                    # Record the comparison result
                    if are_similar:
                        folder_results.append({
                            'Image 1': f'=HYPERLINK("{image_1_path}", "{subfolder.name}/{os.path.basename(image_1_path)}")',
                            'Image 2': f'=HYPERLINK("{image_2_path}", "{other_subfolder.name}/{os.path.basename(image_2_path)}")'
                        })
                    # Append to similarity_data for later visualization
                    similarity_data.append((subfolder.name, other_subfolder.name, ssim_index, pixel_difference, are_similar))
    
    # Return the organized results for this folder
    return folder_results

# Track processing time
start_time = time.time()

# Initialize the set to track compared pairs
compared_pairs = set()

# Use ThreadPoolExecutor to process folders in parallel
with ThreadPoolExecutor() as executor:
    futures = []
    # Submit all folders to be processed
    for subfolder, image_paths in image_paths_dict.items():
        futures.append(executor.submit(compare_images_in_folder, subfolder, image_paths, image_paths_dict, tolerance, ssim_threshold, compared_pairs))
    
    # Collect results in the order they were submitted
    for future in futures:
        try:
            folder_results = future.result()  # Ensure results are collected in the order of submission
            results.extend(folder_results)    # Append results in the correct order
        except Exception as e:
            print(f"An error occurred: {e}")

# Track end time
end_time = time.time()
processing_time = end_time - start_time

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

# Define the red font
red_font = Font(color="FF0000", bold=True)

# Save to Excel with formatting, split into multiple sheets if necessary
output_excel = 'Final_version_Similarity_V4.xlsx'
max_rows_per_sheet = 1048575  # one less than the maximum rows in an Excel sheet to account for the header

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    for start_row in range(0, len(df_results), max_rows_per_sheet):
        end_row = min(start_row + max_rows_per_sheet, len(df_results))
        df_subset = df_results.iloc[start_row:end_row]
        sheet_name = f'Results_{start_row // max_rows_per_sheet + 1}'
        df_subset.to_excel(writer, index=False, sheet_name=sheet_name)
        
        # Get the active worksheet
        worksheet = writer.sheets[sheet_name]
        
        # Apply red font to the rows indicating folder comparison starts
        for row in range(2, len(df_subset) + 2):  # Start from 2 to account for header
            if "Starting comparisons for folder" in str(worksheet.cell(row=row, column=1).value):
                for col in range(1, 3):  # Apply to both columns
                    cell = worksheet.cell(row=row, column=col)
                    cell.font = red_font

print(f"Results saved to {output_excel}")

# Convert similarity data to DataFrame with correct column names
similarity_df = pd.DataFrame(similarity_data, columns=['Folder 1', 'Folder 2', 'SSIM', 'Pixel Difference', 'Is Similar'])

#print(similarity_df)


# Ensure Is Similar column is binary
similarity_df['Is Similar'] = similarity_df['Is Similar'].astype(int)

# Count the number of similar images between each pair of folders
similarity_counts = similarity_df[similarity_df['Is Similar'] == 1].groupby(['Folder 1', 'Folder 2']).size().unstack(fill_value=0)


# Ensure the matrix is symmetrical
similarity_counts = similarity_counts.add(similarity_counts.T, fill_value=0)

# Replace NaN values with 0 for cleaner visualization
similarity_counts = similarity_counts.fillna(0)

# Function to format large numbers
def format_large_numbers(val):
    if val >= 1_000_000:
        return f'{val/1_000_000:.1f}M'
    elif val >= 1_000:
        return f'{val/1_000:.1f}k'
    else:
        return f'{int(val)}'

# Find the pair of folders with the most similar images
max_value = similarity_counts.max().max()
max_pair = similarity_counts.stack().idxmax()

# Adjust figure size based on the number of families
num_families = len(similarity_counts)
fig_size = (max(12, num_families * 0.5), max(12, num_families * 0.5))  # Scale up size, but have a minimum size

# Plot the heatmap with proper counts
#plt.figure(figsize=(12, 10))
# Plot the heatmap
plt.figure(figsize=fig_size)

# Apply the formatting function to each cell's annotation
annot = similarity_counts.applymap(format_large_numbers)
# Plot the heatmap with formatted annotations
ax = sns.heatmap(similarity_counts, annot=annot.values, fmt="", cmap='coolwarm', linewidths=0.5, annot_kws={"size": 6}, cbar_kws={'label': 'Number of Similar Images'})

# Add the "Most Similar Pair" annotation in the bottom-left corner
text_str = f"Most Similar Pair: {max_pair[0]}/{max_pair[1]} = {int(max_value)} similar images"
plt.gcf().text(0.02, 0.02, text_str, fontsize=12, color='black', ha='left', 
               bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.5'))

# Add title and labels
plt.title('Number of Similar Images Between Malware Families', fontsize=16)
plt.xlabel('Malware Family', fontsize=10)
plt.ylabel('Malware Family', fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=6)
plt.yticks(rotation=45, fontsize=6)

# Save and display the heatmap
plt.tight_layout()
plt.savefig('final_similarity_heatmap.png')
plt.show()
#####################
# Threshold for including families in the pie chart individually
threshold_percentage = 1  # Set a threshold of 1%


#pie Count the number of similar images for each family
family_similarities = similarity_df[similarity_df['Is Similar'] == 1].groupby('Folder 1').size()
#pie Calculate total similar images
total_similar_images = family_similarities.sum()
#pie Calculate the percentage of similar images for each family
family_similarities_percentage = (family_similarities / total_similar_images) * 100

# Group smaller sectors into "Other"
large_families = family_similarities_percentage[family_similarities_percentage >= threshold_percentage]
small_families = family_similarities_percentage[family_similarities_percentage < threshold_percentage].sum()
# Add the "Other" category if needed
if small_families > 0:
    large_families['Other'] = small_families

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(family_similarities_percentage, labels=family_similarities.index, autopct='%1.1f%%', startangle=140)

# Add a title
plt.title('Percentage of Similar Images by Malware Family')

# Display the pie chart
plt.savefig('Percentage of Similar Images by Malware Family')
plt.show()
