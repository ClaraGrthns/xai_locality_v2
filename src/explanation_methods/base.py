import numpy as np
import os
from torch.utils.data import DataLoader, Subset
import time 
import torch

def cosine_distance_vectorized(A, B):
    # A.shape: (200, 5)
    # B.shape: (200, kNN, 5)
    dot_products = np.einsum('ik, ijk->ij', A, B)  # Shape: (200, kNN)
    # Compute norms
    A_norms = np.linalg.norm(A, axis=-1)  # Shape: (200,)
    B_norms = np.linalg.norm(B, axis=-1)  # Shape: (200, kNN)
    # Compute cosine similarity
    cosine_similarity = dot_products / (A_norms[:, None] * B_norms)
    
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance

class BaseExplanationMethodHandler:
    def __init__(self,args):
        # self.explainer = self.set_explainer(**kwargs)
        self.args = args
        self.is_smooth_grad = False
    
    def set_explainer(self, **kwargs):
        raise NotImplementedError
    
    def explain_instance(self, **kwargs):
        raise NotImplementedError

    def compute_accuracy(self):
        raise NotImplementedError
    
    def compute_explanations(self):
        raise NotImplementedError
    
    def get_experiment_setting(self):
        raise NotImplementedError
    
    def iterate_over_data(self,
                     tst_dataset, 
                     tst_feat, 
                     analysis_feat, 
                     explanations, 
                     explanations_analysis_set,
                     n_nearest_neighbors,
                     tree,
                     results_path,
                     experiment_setting,
                     results):
        """
        Base method for iterating over data to compute and store metrics.
        """
        chunk_size = 200 #int(np.min((self.args.chunk_size, len(tst_dataset))))
        tst_dataset_loader = DataLoader(tst_dataset, batch_size=chunk_size, shuffle=False)
        for i, batch in enumerate(tst_dataset_loader):
            start = time.time()
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(tst_feat))
            print(f"Processing chunk {i}/{(len(tst_dataset) + chunk_size - 1) // chunk_size}")
            
            explanations_chunk = explanations[chunk_start:chunk_end]
            
            distances, idx_n_neighbors = tree.query(tst_feat[chunk_start:chunk_end], k=n_nearest_neighbors+1, return_distance=True, sort_results=True)
            
            idx_n_neighbors = idx_n_neighbors[:, 1:] # Exclude the first neighbor (itself)
            explanations_n_neighbors = explanations_analysis_set[idx_n_neighbors]

            metrics = self._calculate_distances_kNN(explanations_chunk, explanations_n_neighbors, n_nearest_neighbors)
            metrics["distances"] = distances[:, 1:].T  # Exclude the first distance (itself)
            
            self._update_results_dict(results, metrics, chunk_start, chunk_end)
            
            self._save_chunk_results(results_path, experiment_setting, results)
            
            print(f"Processed chunk {i}/{(len(tst_dataset) + chunk_size - 1) // chunk_size}")
            print(f"Finished processing chunk {i} in {time.time() - start:.2f} seconds")

        return results
    
    def _calculate_distances_kNN(self, explanations_test, explanations_analysis_kNN, n_nearest_neighbors):
        """
        Calculate various metrics from chunk results.
        
        Args:
            explanations_test: Explanations for the test set, shape (num_test_samples, num_features)
            explanations_analysis_kNN: Explanations for the k nearest neighbors in the analysis set, shape (num_test_samples, n_nearest_neighbors, num_features)
            n_nearest_neighbors: Number of nearest neighbors to consider
            
        Returns:
            Dictionary of calculated metrics
        """
        differences = explanations_test[:, None, :] - explanations_analysis_kNN # Shape: (num_test_samples, n_nearest_neighbors, num_features)
        l1_distances = np.linalg.norm(differences, ord=1, axis=-1) # Shape: (num_test_samples, n_nearest_neighbors)
        l2_distances = np.linalg.norm(differences, ord=2, axis=-1)
        cosine_distances =  cosine_distance_vectorized(explanations_test, explanations_analysis_kNN)
        
        kNNs_for_mean = np.arange(1, n_nearest_neighbors + 1)[None, :]  # Shape: (1, n_nearest_neighbors)
        l1_distance_cumsum = np.cumsum(l1_distances, axis=1) / kNNs_for_mean # Shape: (num_test_samples, n_nearest_neighbors)
        l2_distance_cumsum = np.cumsum(l2_distances, axis=1) / kNNs_for_mean
        cosine_distance_cumsum = np.cumsum(cosine_distances, axis=1) / kNNs_for_mean

        return {
            "l1_distance_accumulated": l1_distance_cumsum.T,
            "l2_distance_accumulated": l2_distance_cumsum.T,
            "cosine_distance_accumulated": cosine_distance_cumsum.T,
            "l1_distance": l1_distances.T,
            "l2_distance": l2_distances.T,
            "cosine_distance": cosine_distances.T,
        }
       
        
    
    def _update_results_dict(self, results, metrics, chunk_start, chunk_end):
        """
        Update the results dictionary with calculated metrics.
        
        Args:
            results: Dictionary to update
            metrics: Dictionary of calculated metrics
            chunk_start: Starting index of current chunk
            chunk_end: Ending index of current chunk
        """
        for key, value in metrics.items():
            if key in results:
                results[key][:, chunk_start:chunk_end] = value
    
    def _save_chunk_results(self, results_path, experiment_setting, results):
        """
        Save results to disk.
        
        Args:
            results_path: Path to save results
            experiment_setting: Experiment setting identifier
            results: Results dictionary to save
        """
        # np.savez(os.path.join(results_path, experiment_setting), **results)
        if self.args.create_additional_analysis_data:
           results_path = os.path.join(results_path, "downsampled")
           if not os.path.exists(results_path):
               os.makedirs(results_path)
           np.savez(os.path.join(results_path, experiment_setting), **results)
        else:
           np.savez(os.path.join(results_path, experiment_setting), **results)
    

    
    def update_results(self, results, idx, chunk_start, chunk_end, metrics):
        """Update the results dictionary with the computed metrics."""
        for key, value in metrics.items():
            if key in results:
                results[key][idx, chunk_start:chunk_end] = value
        return results
    
    def set_experiment_setting(self, n_nearest_neighbors):
        self.experiment_setting = self.get_experiment_setting(n_nearest_neighbors)
        return self.experiment_setting
    
    def run_analysis(self, 
                     tst_dataset, 
                     tst_feat, 
                     analysis_feat,
                     explanations, 
                     explanations_analysis_set,
                     n_nearest_neighbors, 
                     tree,
                     results_path,
                     ):
        range_n_nearest_neighbors = np.arange(n_nearest_neighbors)
        
        experiment_setting = self.experiment_setting
        
        results = {
            "distances": np.full((n_nearest_neighbors, self.args.max_test_points), np.nan),
            "l1_distance": np.full((n_nearest_neighbors, self.args.max_test_points), np.nan),
            "l2_distance": np.full((n_nearest_neighbors, self.args.max_test_points), np.nan),
            "cosine_distance": np.full((n_nearest_neighbors, self.args.max_test_points), np.nan),
            "l1_distance_accumulated": np.full((n_nearest_neighbors, self.args.max_test_points), np.nan),
            "l2_distance_accumulated": np.full((n_nearest_neighbors, self.args.max_test_points), np.nan),
            "cosine_distance_accumulated": np.full((n_nearest_neighbors, self.args.max_test_points), np.nan),
            "n_nearest_neighbors": range_n_nearest_neighbors,
            "analysis_set_size": len(explanations_analysis_set),
        }

        results = self.iterate_over_data(
                tst_dataset=tst_dataset,
                tst_feat=tst_feat,
                analysis_feat=analysis_feat,
                explanations=explanations,
                explanations_analysis_set=explanations_analysis_set,
                n_nearest_neighbors=n_nearest_neighbors, 
                tree=tree,
                results_path=results_path,
                experiment_setting=experiment_setting,
                results=results)
        return results