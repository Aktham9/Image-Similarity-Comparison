# Malware Mutation Detection

This repository contains a script designed to identify mutations in malware by comparing images representing different malware families. The script compares images from different folders (each representing a malware family) to find similar images, indicating potential mutations.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Malware evolves over time, and identifying mutations across different malware families is crucial for cybersecurity. This project provides a method to detect these mutations by comparing images that represent malware samples.

## Features
- Converts images to black and white for comparison.
- Compares images using Structural Similarity Index (SSIM) and pixel difference.
- Excludes identical images.
- Generates various visualizations to represent the similarity findings.
- Outputs the results to an Excel file with clear formatting.

## Installation
To use this script, you need to have Python and the required libraries installed. You can install the dependencies using `pip`.

```bash
pip install numpy pandas matplotlib seaborn pillow scikit-image openpyxl tqdm
