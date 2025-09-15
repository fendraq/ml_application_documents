import ijson
import polars as pl

def load_json_section(filename, section_key):
    with open(filename, 'rb') as file:
        items = ijson.items(file, f'{section_key}.item')
        return list(items)
    
def load_json_section_chunked(filename, section_key, chunk_size=10000):
    chunks = []
    current_chunk = []

    with open(filename, 'rb') as file:
        items = ijson.items(file, f'{section_key}.item')

        for item in items:
            current_chunk.append(item)

            if len(current_chunk) >= chunk_size:
                chunk_df = pl.DataFrame(current_chunk)
                chunks.append(chunk_df)
                current_chunk = []
                print(f"Processed chunk {len(chunks)} for {section_key}")

        if current_chunk:
            chunk_df = pl.DataFrame(current_chunk)
            chunks.append(chunk_df)

    return pl.concat(chunks)

print("Loading images...")
images_data = load_json_section('../data/Pub_Lay_Net/labels.json', 'images')
df_images = pl.DataFrame(images_data)

print("Loading categories...")
categories_data = load_json_section('../data/Pub_Lay_Net/labels.json', 'categories')
df_categories = pl.DataFrame(categories_data)

print("Loading annotations...")
annotations_data = load_json_section_chunked('../data/Pub_Lay_Net/labels.json', 'annotations')
df_annotations = pl.DataFrame(annotations_data)


print("Merging step 1")
df_merged = df_annotations.join(
    df_images, 
    left_on='image_id',
    right_on='id',
    suffix='_img'
)

print("Merging step 2")
df_full = df_merged.join(
    df_categories, 
    left_on='category_id',
    right_on='id',
    suffix='_cat'
)

print("Load complete")
print(f"DataFrame Head: {df_full.head()}")

print("Saving DataFrame with parquet")
df_images.write_parquet('../data/processed/labels/df_images.parquet')
df_annotations.write_parquet('../data/processed/labels/df_annotations.parquet')
df_categories.write_parquet('../data/processed/labels/df_categories.parquet')
df_full.write_parquet('../data/processed/labels/df_full.parquet')

print(f"Saving done! Full dataset shape: {df_full.shape}")
