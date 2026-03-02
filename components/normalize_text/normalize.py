import argparse
import os
import pandas as pd
import re

def clean_text(text):
    # 1. Catch missing/null values and turn them into empty strings
    if not isinstance(text, str):
        return ""
    
    # 2. Lowercase the text
    text = text.lower()
    
    # 3. Replace URLs with the word 'URL' (Regex looks for http or www)
    text = re.sub(r'http\S+|www\.\S+', 'URL', text)
    
    # 4. Replace numbers with the word 'NUM' (Regex \d+ looks for digits)
    text = re.sub(r'\d+', 'NUM', text)
    
    # 5. Remove punctuation (Regex [^\w\s] removes anything that isn't a word or space)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 6. Trim extra whitespace (removes double spaces and leading/trailing spaces)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    # Set up the inputs and outputs that Azure ML will pass to the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # Load the dataset using Pandas
    print("Loading data...")
    df = pd.read_parquet(args.data)

    # Assuming the column containing the review text is 'reviewText' 
    # (This is standard for the Amazon Electronics dataset)
    text_column = 'reviewText' 

    # Apply the clean_text function we wrote above to every row in the column
    print("Normalizing text (this might take a moment)...")
    df[text_column] = df[text_column].apply(clean_text)

    # Filter out empty or very short reviews (<10 characters)
    initial_count = len(df)
    df = df[df[text_column].str.len() >= 10]
    final_count = len(df)
    
    print(f"Dropped {initial_count - final_count} short/empty reviews.")
    print(f"Remaining rows: {final_count}")

    # Write the cleaned data back out as a parquet file
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))

if __name__ == "__main__":
    main()