from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class SimilarityMeasure(ABC):
    """
    Abstract class to represent a similarity measure method
    
    Methods
    -------
    compute_similarity(v1, v2)
        Calculate similarity between v1 and v2 
    """
    
    @abstractmethod
    def compute_similarity(self, v1, v2):
        """Calculate similarity between v1 and v2 using a similarity metrics

        Parameters
        ----------
        v1: numpy array
            one dimensional vector
        v2: numpy array 
            one dimensional 

        Returns
        -------
        float
            similarity value between v1 and v2. 
            The higher the value, the higher the similarity between v1 and v2,
            so several modifications needed for metrics that doesn't follow that rule 
        """
        
        pass

class CosineSimilarity(SimilarityMeasure):
    """
    Concrete class to represent a similarity measure method using Cosine Similarity
    """
    
    def compute_similarity(self, v1, v2):
        return cosine_similarity(v1, v2)

class EuclideanDistance(SimilarityMeasure):
    """
    Concrete class to represent a similarity measure method using Euclidean Distance
    """
    
    def compute_similarity(self, v1, v2):
        return -1*euclidean_distances(v1, v2)

class JaccardCoefficient(SimilarityMeasure):
    """
    Concrete class to represent a similarity measure method using Jaccard Coefficient
    """
    
    def compute_similarity(self, v1, v2):
        lst_1 = v1[0]
        lst_2 = v2[0]
        intersection = len(list(set(lst_1).intersection(lst_2)))
        union = (len(lst_1) + len(lst_2)) - intersection
        return float(intersection) / union