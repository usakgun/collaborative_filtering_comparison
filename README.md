# Collaborative Filtering Recommender System

This project implements and compares two fundamental Memory-Based Collaborative Filtering techniques: User-Based and Item-Based approaches using the MovieLens 100k dataset.

## Project Overview
The goal is to analyze the trade-offs between user-centric and item-centric recommendation algorithms. The system calculates similarity matrices to predict user ratings and evaluates the accuracy using Root Mean Squared Error (RMSE).

## Methodology
The project explores two distinct strategies:
1.  **User-Based Collaborative Filtering:** Recommends items by identifying users with similar rating patterns.
2.  **Item-Based Collaborative Filtering:** Recommends items that are similar to those a user has liked in the past.

Both methods utilize **Cosine Similarity** to establish relationships between users or items.

## Performance Analysis
The algorithms were evaluated on a test set (25% of the data). The Item-Based approach demonstrated slightly better performance and stability for this specific dataset.

* **User-Based CF RMSE:** 3.3970
* **Item-Based CF RMSE:** 3.3906

## Technologies
* Python
* Pandas
* Scikit-Learn (Pairwise Distances, Mean Squared Error)
* NumPy
