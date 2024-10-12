from sklearn.neighbors import BallTree
import numpy as np

class SSSD:
    def __init__(self, X, scores, k=5, m=5):
        """
        Initialize the SSSD class with input data X anomly scores scores, number of neighbors k and window size m.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Input data points.
        scores : array-like, shape (n_samples,)
            Anomaly scores associated with the data points.
        k : int, optional (default=5)
            Number of nearest neighbors.
        m : int, optional (default=5)
            Window size for moving average.
        """
        self.X = X
        self.tree = BallTree(X)
        self.n_samples = X.shape[0]
        
        self.scores = scores
        self.index = scores.argsort()
        
        self.k = k
        self.m = m
        # Create a BallTree for fast nearest-neighbor lookup
        

    # Static method for two-sided moving average
    @staticmethod
    def MoveAverageTwoSides(y, m=5):
        """
        Compute a two-sided moving average for a given array.
        
        Parameters:
        y : array-like
            Input data array.
        m : int, optional (default=5)
            Window size for moving average.
        
        
        Returns:
        Mt : list
            Moving average smoothed values.
        """
        Mt = []

        # Loop through each element in the input data
        for i in range(len(y)):
            # Determine window boundaries
            start = max(0, i - m)
            end = min(len(y), i + m)

            # Calculate mean for the window and append to the result list
            Mt.append(y[start:end].mean())
            
        return Mt

    
     # Method to compute Smoothed Subspace Scoring Difference (SSSD)
    def getSSSD(self, sortedNearbyMetric):
        """
        Calculate the SSSD

        Parameters:
        sortedNearbyMetric : array-like
            Sorted nearby metrics (e.g., distance or anomaly scores).

        Returns:
        SSSD : array-like
            SSSD for each data.
        SSSDsum : float
            Sum of the SSSD for all the data.
        """
        sortedNearbyMetric = (sortedNearbyMetric - np.min(sortedNearbyMetric)) / (np.max(sortedNearbyMetric) - np.min(sortedNearbyMetric))
        averageNearbyMetric = self.MoveAverageTwoSides(sortedNearbyMetric, self.m)
        diffmetric = abs(sortedNearbyMetric - averageNearbyMetric)
        T = self.MoveAverageTwoSides(diffmetric, self.m)
        diffmetric = diffmetric - T
        SSSD = np.where(diffmetric > 0, diffmetric, 0)
        index = np.argsort(-SSSD)
        SSSDsum = np.sum(SSSD)

        return SSSD, SSSDsum


    def compute_nearby_scores(self):
        """
        Compute the nearby scores for all points based on k-nearest neighbors.
        
        Returns:
        nearbyScore : array-like
            Computed nearby scores for each data point.
        index : array-like
            Sorted index of the input scores.
        """
        nearbyScore = np.zeros((self.n_samples,))

        # Compute nearby scores for each data point
        for m, item in enumerate(self.X):
            dist, ind = self.tree.query(item.reshape(1, -1), self.k + 1)
            itemscore = 0
            for j in range(self.k):
                itemscore += self.scores[ind[0][j+1]] * (1 - j / (self.k * 2))
            nearbyScore[m] = itemscore

        return nearbyScore
    
    def compute_nearby_distances(self):
        """
        Compute the nearby scores for all points based on k-nearest neighbors.
        
        Returns:
        nearbyScore : array-like
            Computed nearby scores for each data point.
        index : array-like
            Sorted index of the input scores.
        """
        nearbyDistance = np.zeros((self.n_samples,))

        # Compute nearby scores for each data point
        for m, item in enumerate(self.X):
            dist, ind = self.tree.query(item.reshape(1, -1), self.k + 1)
            itemscore = 0
            for j in range(self.k):
                itemscore += dist[0][j+1] * (1 - j / (self.k * 2))
            nearbyDistance[m] = itemscore

        return nearbyDistance
    

    def get_Score_SSSD(self):
        """
        Compute Score_SSSD.
        
        
        Returns:
        SSSD : array-like
            Score_SSSD for each data.
        SSSDsum : float
            Sum of SSSD values for all the data.
        misslabeledindex : array-like
            Indices of potential mislabeled points detected by Score_SSSD(index with higher probability of mislabeling is smaller).
        """
        nearbyScore = self.compute_nearby_scores()
        sortedNearbyScore = nearbyScore[self.index]
        SSSD, SSSDsum = self.getSSSD(sortedNearbyScore)
        misslabeledindex = self.index[np.argsort(-SSSD)]

        original_order = np.argsort(self.index)
        SSSD = SSSD[original_order]

        return SSSD, SSSDsum, misslabeledindex

    def get_Distance_SSSD(self):
        """
        Compute Distance_SSSD.
         
        Returns:
        SSSD : array-like
            Distance_SSSD for each data.
        SSSDsum : float
            Sum of Distance_SSSD values for all the data.
        misslabeledindex : array-like
            Indices of potential mislabeled points detected by Distance_SSSD(index with higher probability of mislabeling is smaller).
        """
        nearbyDistance = self.compute_nearby_distances()
        sortedNearbyDistance = nearbyDistance[self.index]
        SSSD, SSSDsum = self.getSSSD(sortedNearbyDistance)
        
        misslabeledindex = self.index[np.argsort(-SSSD)]

        original_order = np.argsort(self.index)
        SSSD = SSSD[original_order]

        return SSSD, SSSDsum, misslabeledindex

    def get_final_SSSD(self):
        """
        Compute the final SSSD by summing Score_SSSD and Distance_SSSD.
        
        Parameters:
        scores : array-like, shape (n_samples,)
            Scores associated with the data points.
        
        Returns:
        final_SSSD : array-like
            Combined SSSD scores.
        final_SSSD_sum : float
            Sum of the final SSSD scores.
        """
        Score_SSSD, _, _ = self.get_Score_SSSD()
        Distance_SSSD, _, _ = self.get_Distance_SSSD()
        
        # Combine Score_SSSD and Distance_SSSD
        final_SSSD = (Score_SSSD + Distance_SSSD)/2.0
        final_SSSD_sum = np.sum(final_SSSD)

#         youwenti
        misslabeledindex = np.argsort(-final_SSSD)
        
        return final_SSSD, final_SSSD_sum, misslabeledindex
