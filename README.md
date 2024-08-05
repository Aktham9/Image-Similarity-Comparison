# Malware Mutation Detection
##Overview
This repository contains a Python script for detecting mutations in malware families by comparing images representing different malware samples. The script uses Structural Similarity Index (SSIM) and pixel difference to identify similar images across different malware families.

##Features
Converts images to black and white for standardized comparison.
Uses SSIM and pixel difference to detect similarity while excluding identical images.
Compares images across different malware families (folders) only.
Outputs the results in an Excel file, indicating the start of comparisons for each folder with red font.
Provides various visualizations to interpret the results, including:
Pie Chart of Similar vs. Dissimilar Samples
Histogram of SSIM Scores
Box Plot of SSIM Scores
Bar Chart of Similarity Counts per Malware Family
