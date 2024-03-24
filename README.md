# Recommendation Systems in Python

In this project, we compared the results of different recommendation systems using the [Surprise](https://github.com/NicolasHug/Surprise) library based on the [MovieLens Small](https://grouplens.org/datasets/movielens/latest/) dataset.

The recommendation systems used are:

- NormalPredictor
- SVD
- SVD++
- NMF
- KNNBasic
- KNNWithMeans
- KNNWithZScore
- KNNBaseline
- SlopeOne
- CoClustering

Additionally, custom recommendation systems have been implemented, such as:

- KNNWithMeansWeighted, where the average ratings per user are calculated through weighted averaging. The weights used are represented by the timestamps of individual views
- BinaryPredictor, where the values are converted from the scale $[0, 5]$ to the scale $[0, 1]$, using the mean rating value as the threshold

The complete report, which includes a comprehensive analysis of the data and a detailed explanation of each recommendation system, is available as a PDF inside the repository. The report is written in Italian.

