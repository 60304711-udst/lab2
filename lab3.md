 


Amazon Electronics Data Pipeline lab3
Name: Mohammad Shurbaji
ID: 60304711
1. Project Overview
This project implements a production-grade data engineering pipeline using the Medallion (Lakehouse) Architecture. The goal is to process raw Amazon Electronics review data and product metadata into a curated "Gold" dataset ready for Machine Learning and Business Intelligence.
2. Technology Stack
•	Azure Databricks: A managed Apache Spark environment used for distributed data processing.
•	PySpark: The Python API for Spark, used to perform transformations on large-scale DataFrames.
•	Azure Data Lake Storage (ADLS Gen2): The storage layer where data is organized into containers (raw, processed, curated).
•	Parquet: An optimized columnar storage format that provides better performance and compression than JSON.
•	Databricks Jobs: An orchestration tool used to automate the execution of the notebooks in a specific sequence.
3. The ETL Pipeline & Notebook Mapping
The pipeline follows the Extract, Transform, Load (ETL) pattern, mapped across three notebooks:
Notebook 1: Load and Clean (Extract & Transform)
•	Mapping: Represents the Bronze to Silver transition.
•	Process: Extracts raw JSON data, removes null values in critical fields (asin, reviewerID, overall), trims review text, and filters out reviews shorter than 10 characters .
•	Output: Cleaned reviews saved as Parquet in the processed container .
Notebook 2: Enrich with Metadata (Transform)
•	Mapping: Represents the Silver refinement stage.
•	Process: Joins the cleaned reviews with product metadata (Title, Brand, Price) using the asin key .
•	Output: An enriched dataset that combines customer feedback with product-specific details.
Notebook 3: Write Gold Features (Load)
•	Mapping: Represents the Silver to Gold transition.
•	Process: Selects the final 11 features required for the data model .
•	Output: The final curated dataset saved in the curated container, ready for analysis .
4. Homework & Reflections
Mapping Notebooks to ETL
As detailed above, Notebook 1 handles Extraction and initial Cleaning, Notebook 2 performs the Enrichment (Complex Transformation), and Notebook 3 executes the final Load into the Gold layer for downstream consumption.
Potential Further Enrichments
To improve the Gold layer, we could:
•	Sentiment Analysis: Use NLP libraries to calculate a "Sentiment Score" from the reviewText.
•	Helpfulness Ratio: Create a feature that calculates the percentage of helpful votes vs total votes.
•	Currency Conversion: Convert prices to Qatari Riyals (QAR) for local market analysis.
1.	Rating Distribution: A bar chart showing the frequency of each rating (1-5).
o	Insight: This shows if users are generally satisfied or if there is a bias toward extreme ratings.
2.	Average Rating by Brand: A comparison of the top 10 brands.
o	Insight: Identifies which electronics brands are performing best according to customer feedback.

<img width="940" height="431" alt="image" src="https://github.com/user-attachments/assets/95b557c4-b24b-4ca0-b567-be529a3bb405" />
