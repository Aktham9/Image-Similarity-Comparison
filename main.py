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

    return ssim_index, pixel_difference, ssim_index > ssim_threshold and pixel_difference <= tolerance

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

# Function to compare images in a folder and subsequent folders
def compare_images_in_folder(subfolder, image_paths, all_features_dict, tolerance, ssim_threshold):
    folder_results = []
    compared_pairs = set()


    # Add a row indicating the start of comparisons for the current folder
    folder_results.append({'Image 1': f'Starting comparisons for folder: {subfolder.name}', 'Image 2': ''})
    
    for i in tqdm(range(len(image_paths)), desc=f"Processing {subfolder.name} folder", ncols=100):
        image_1_path = str(image_paths[i])
        
        # Compare with images in the same folder
        for j in range(i + 1, len(image_paths)):
            image_2_path = str(image_paths[j])
            pair = tuple(sorted([image_1_path, image_2_path]))
            if pair not in compared_pairs:
                ssim_index, pixel_difference, are_similar = compare_images(image_1_path, image_2_path, tolerance, ssim_threshold)
                similarity_data.append((subfolder.name, subfolder.name, ssim_index, pixel_difference, are_similar))
                if are_similar:
                    folder_results.append({
                        'Image 1': f'=HYPERLINK("{image_1_path}", "{subfolder.name}/{os.path.basename(image_1_path)}")',
                        'Image 2': f'=HYPERLINK("{image_2_path}", "{subfolder.name}/{os.path.basename(image_2_path)}")'
                    })
                compared_pairs.add(pair)
        
        # Compare with images in subsequent folders
        for other_subfolder, other_image_paths in all_features_dict.items():
            if other_subfolder == subfolder:
                continue
            for k in range(len(other_image_paths)):
                image_2_path = str(other_image_paths[k])
                pair = tuple(sorted([image_1_path, image_2_path]))
                if pair not in compared_pairs:
                    ssim_index, pixel_difference, are_similar = compare_images(image_1_path, image_2_path, tolerance, ssim_threshold)
                    similarity_data.append((subfolder.name, other_subfolder.name, ssim_index, pixel_difference, are_similar))
                    if are_similar:
                        folder_results.append({
                            'Image 1': f'=HYPERLINK("{image_1_path}", "{subfolder.name}/{os.path.basename(image_1_path)}")',
                            'Image 2': f'=HYPERLINK("{image_2_path}", "{other_subfolder.name}/{os.path.basename(image_2_path)}")'
                        })
                    compared_pairs.add(pair)
    
    return folder_results

# Track processing time
start_time = time.time()

# Use ThreadPoolExecutor to process folders in parallel
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(compare_images_in_folder, subfolder, image_paths, image_paths_dict, tolerance, ssim_threshold): subfolder for subfolder, image_paths in image_paths_dict.items()}
    for future in as_completed(futures):
        folder_results = future.result()
        results.extend(folder_results)

# Track end time
end_time = time.time()
processing_time = end_time - start_time

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

# Define the red font
red_font = Font(color="FF0000", bold=True)

# Save to Excel with formatting, split into multiple sheets if necessary
output_excel = 'Final_version_Similarity.xlsx'
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

# Convert similarity data to DataFrame
similarity_df = pd.DataFrame(similarity_data, columns=['Folder 1', 'Folder 2', 'SSIM', 'Pixel Difference', 'Is Similar'])

# Visualizations

# 1. Pie Chart of Similar vs. Dissimilar Samples
similar_count = similarity_df['Is Similar'].sum()
dissimilar_count = len(similarity_df) - similar_count
plt.figure(figsize=(8, 8))
plt.pie([similar_count, dissimilar_count], labels=['Similar', 'Dissimilar'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
plt.title('Proportion of Similar vs. Dissimilar Samples')
plt.savefig('pie_chart_similarity.png')
plt.show()

# 2. Histogram of SSIM Scores
plt.figure(figsize=(10, 6))
sns.histplot(similarity_df['SSIM'], kde=True, bins=30)
plt.title('Distribution of SSIM Scores')
plt.xlabel('SSIM Score')
plt.ylabel('Frequency')
plt.savefig('histogram_ssim.png')
plt.show()

# 3. Box Plot of SSIM Scores
plt.figure(figsize=(10, 6))
sns.boxplot(x='Is Similar', y='SSIM', data=similarity_df)
plt.title('Box Plot of SSIM Scores')
plt.xlabel('Similarity')
plt.ylabel('SSIM Score')
plt.savefig('boxplot_ssim.png')
plt.show()

# 4. Bar Chart of Similarity Counts per Folder (Family)
folder_similar_counts = similarity_df[similarity_df['Is Similar']].groupby('Folder 1').size()
plt.figure(figsize=(12, 8))
folder_similar_counts.plot(kind='bar')
plt.title('Number of Similar Image Pairs per Folder (Family)')
plt.xlabel('Family')
plt.ylabel('Count of Similar Pairs')
plt.savefig('bar_chart_folder_similarity.png')
plt.show()

# 5. Bar Chart of Similarity Counts within Same Folder vs. Different Folders
same_folder_count = similarity_df[(similarity_df['Is Similar']) & (similarity_df['Folder 1'] == similarity_df['Folder 2'])].shape[0]
different_folder_count = similarity_df[(similarity_df['Is Similar']) & (similarity_df['Folder 1'] != similarity_df['Folder 2'])].shape[0]
plt.figure(figsize=(8, 6))
plt.bar(['Same Family', 'Different Families'], [same_folder_count, different_folder_count], color=['#66b3ff','#ff9999'])
plt.title('Similarity Counts within Same Family vs. Different Families')
plt.xlabel('Comparison Type')
plt.ylabel('Count of Similar Pairs')
plt.savefig('bar_chart_same_vs_different_folders.png')
plt.show()

# 6. Line Chart of Processing Time
plt.figure(figsize=(10, 6))
plt.plot([0, len(similarity_df)], [0, processing_time], marker='o')
plt.title('Processing Time')
plt.xlabel('Number of Image Comparisons')
plt.ylabel('Time (seconds)')
plt.savefig('line_chart_processing_time.png')
plt.show()

# 7. Scatter Plot of Pixel Differences vs. SSIM Scores
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Pixel Difference', y='SSIM', hue='Is Similar', data=similarity_df)
plt.title('Pixel Differences vs. SSIM Scores')
plt.xlabel('Pixel Difference')
plt.ylabel('SSIM Score')
plt.savefig('scatterplot_pixel_difference_ssim.png')
plt.show()
