# Basketball-with-Machine-Learning

This project applies unsupervised learning to identify the very best basketball players to play in the NBA from 1983-2003. Additionally, a variety 
of supervised learning algorithms are used to perform NBA game outcome prediction, and a comparison is performed.

We find the top 87 players, including players such as Michael Jordan, Larry Bird, and Kobe Bryant, and we also see that of the models tested, the AdaBoost
ensemble classifier performs the best at predicting the outcome of NBA basketball games, with an accuracy of 62.95% on the testing data.

Breakdown of files:
      
    Clustering.py - this contains the code to run the k-means clustering algorithm on the player season data. It performs the clustering, selects the clusters
                    representing the best players, and then runs the clustering algorithm on those clusters to further filter for the very best players.
    
    
