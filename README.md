# Video Screening Pipeline

An AI-assisted video screening pipeline for identifying relevant **Infant & Child CPR and Choking Management** videos from YouTube and TikTok.

The project combines video metadata collection, preprocessing, manual annotation, keyword-based screening, machine learning, confidence-based filtering, and exploratory data analysis (EDA) to identify videos relevant to pediatric CPR and choking management.

---

## Overview

The purpose of this project is to build a reproducible pipeline for screening large numbers of online videos related to **Infant & Child CPR and Choking Management**.

The pipeline collects videos from YouTube and TikTok using multiple topic-specific search queries, cleans and deduplicates the collected data, creates a manually labeled sample, trains and evaluates text-based screening models, and applies the selected approach to the full dataset.

The final dataset is then analyzed using exploratory data analysis to understand platform distribution, engagement, upload trends, languages, creators, subtopics, and other characteristics.

---

## Aim

The main aim of this project is to develop a systematic and reproducible method for identifying videos that are relevant to **Infant & Child CPR and Choking Management**.

The project focuses on reducing manual screening effort while maintaining a transparent and auditable screening process.

---

## Key Features

* YouTube and TikTok video collection
* Multiple topic-specific search queries
* Metadata extraction
* Data cleaning and normalization
* Video deduplication
* Manual relevance annotation
* Explicit relevance criteria
* Keyword-based screening
* TF-IDF based machine learning
* Multilingual sentence embeddings
* Logistic Regression classification
* Model evaluation using precision, recall, F1-score, and confusion matrix
* Confidence-based video classification
* Audit set for uncertain predictions
* Platform-level analysis
* Creator concentration analysis
* Subtopic analysis
* Language analysis
* Three-stage exploratory data analysis
* Robustness analysis using confident predictions
* Creator contribution sensitivity analysis
* Reproducible reports, tables, and figures

---

## Data Collection

The project uses search queries covering major aspects of pediatric CPR and choking management, including:

* Pediatric CPR
* Infant CPR
* Child CPR
* Pediatric BLS
* Choking management
* Infant choking
* Child choking
* Airway obstruction
* Chest compressions
* Rescue breathing
* AED
* Pediatric resuscitation
* Neonatal resuscitation
* PALS
* Related CPR and choking combinations

A total of **35 search queries** were used across YouTube and TikTok.

---

## Key Findings

The final relevant dataset contains **1,309 videos**.

Platform distribution:

* **TikTok:** 56.0%
* **YouTube:** 44.0%

Other major observations include:

* English was the dominant detected language.
* Choking and CPR were the most common subtopic signals.
* Most creators contributed only one video.
* Creator concentration was relatively limited, with the top 10 creators accounting for approximately 14.5% of the main relevant dataset.
* The major subtopic and language patterns remained broadly stable in the confident subset.
* The main findings also remained broadly stable after applying the creator contribution cap.

These robustness checks increase confidence that the overall EDA patterns are not driven only by uncertain predictions or a small number of highly active creators.

---
