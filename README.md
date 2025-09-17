# Exploring Pub Lay Net Assignment

## Purpose
This project is an exploration of how to prepare a large dataset for modeltraining with the purpose of identifying title pages and text elements. 
## Steps taken
1. pln_data_preprocessing.py -> reading and converting json to DF
2. pln_delete_unused_data.ipynb -> keeping rows matching document files
3. pln_add_title_page.ipynb -> adding a new column for title_page = True/False
4. pln_title_df.ipynb -> removing all columns except file_name and title_page
5. pln_image_downscaling.ipynb -> Preprocessed images for faster training
6. pln_mini_batch.ipynb -> Extracting 500 observations for exploratory model training with GridSearchCV
7. pln_exploratory_model_training.ipynb -> Exploatory modeltraining av RandomForestClassifier, KNeighborsClassifier, SVC and LogistivRegression



## Data Frames
- df_annotations.parquet (Original annotations data)
- df_categories.parquet (Original categories data)
- df_images.parquet (Original images data)
- df_full.parquet (Original dataset with all labels (annotations, categories and images) merged)
- df_full_filtered.parquet (Original dataset but filtered to images from Kaggle) 
- df_full_filtered_with_title.parquet (Original dataset with added column title_page (True/False))
- df_title_page_classification.parquet (Only with unique file_name and title_page columns)

## Processed images
- pln_X_features_raw_128x128.npy - Array of the images, downscaled
- pln_y_labels.npy - labels for downscaled images.