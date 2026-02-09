# lab2
Amazon Reviews ETL Pipeline
Project Overview
This project demonstrates a scalable End-to-End ETL (Extract, Transform, Load) pipeline built on Microsoft Azure. The goal was to process a large dataset of Amazon Electronics reviews (JSON format), convert them into an optimized storage format (Parquet), and partition the data by year for efficient downstream analysis.

Architecture
The pipeline consists of the following components:

Storage: Azure Data Lake Storage Gen2 (ADLS Gen2) using three containers: raw, processed, and curated.

Ingestion: Big data files were moved from a Virtual Machine to the raw container.

Orchestration & Transformation: Azure Data Factory (ADF) was used to build a Mapping Data Flow for data cleaning and partitioning.

Transformation Logic
The core logic was performed within a Mapping Data Flow:

Source: Ingested raw JSON files from the raw container.

Derived Column: Created a new column named reviewYear.

Expression: year(toTimestamp(toLong(unixReviewTime) * 1000))

Sink: Exported the data to the processed container in Parquet format.

Optimization: Used Key Partitioning on the reviewYear column to organize data into year-specific folders.

Challenges & Troubleshooting
Issue: Encountered a validation error where the Sink transformation could not "see" the newly created reviewYear column.

Resolution: Disabled Auto-mapping and implemented Fixed Mapping to manually link the stream to the output. This ensured the schema was correctly recognized for partitioning.

Evidence of Completion:
<img width="940" height="429" alt="image" src="https://github.com/user-attachments/assets/17740d2a-1055-4ef6-ab49-e51e485bd8a9" />
<img width="940" height="433" alt="image" src="https://github.com/user-attachments/assets/9b3cee43-6eda-41a3-9374-38a45d091e70" />


