import numpy as np

def uniform_ball_sample(centers, R_list, N_per_center, distance_measure="euclidean", pad_nan=True, random_seed=42):
    """
    Uniformly sample N_per_center points in balls around each center for each radius in R_list.
    Optionally pad with NaN to create a triangular structure.
    
    Args:
        centers (np.ndarray): The centers of the balls, shape (N_samples, dim_feat).
        R_list (np.ndarray or list): The radii of the balls, shape (N_radii,).
        N_per_center (int): The number of points to sample per center per radius.
        distance_measure (str): The distance measure to use ("euclidean" or "manhattan").
        pad_nan (bool): If True, pads samples with NaN to create a triangular structure.
        random_seed (int, optional): Seed for reproducible random sampling.
        
    Returns:
        np.ndarray: 
            - If pad_nan=False: shape (N_samples, N_radii, N_per_center, dim_feat).
            - If pad_nan=True: shape (N_samples, N_radii, N_per_center * N_radii, dim_feat).
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    N_samples, dim_feat = centers.shape
    N_radii = len(R_list)
    
    centers_reshaped = centers.reshape(N_samples, 1, 1, dim_feat)
    if distance_measure == "euclidean":
        # Sample random directions (normalized), shape (N_samples, N_radii, N_per_center, dim_feat)
        directions = np.random.normal(size=(N_samples, N_radii, N_per_center, dim_feat))
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        # Sample distances, shape (N_samples, N_radii, N_per_center)
        distances = R_list.reshape(1, N_radii, 1) * np.random.uniform(
            0, 1, size=(N_samples, N_radii, N_per_center))
        distances = distances ** (1/dim_feat)
        samples = centers_reshaped + directions * distances[..., np.newaxis]
    elif distance_measure == "manhattan":
        # Generate random directions (exponential with random signs)
        z = np.random.exponential(scale=1.0, size=(N_samples, N_radii, N_per_center, dim_feat))
        z *= np.random.choice([-1, 1], size=(N_samples, N_radii, N_per_center, dim_feat))
        # Project onto L1-sphere (normalize each row)
        u = z / np.linalg.norm(z, ord=1, axis=-1, keepdims=True)
        # Scale by random radii
        radii = R_list.reshape(1, N_radii, 1) * np.random.uniform(
            0, 1, size=(N_samples, N_radii, N_per_center))
        radii = radii ** (1/dim_feat)
        samples = centers_reshaped + u * radii[..., np.newaxis]
    else:
        raise ValueError("Invalid distance_measure. Use 'euclidean' or 'manhattan'.")
    if pad_nan:
        # Create a NaN-padded array of shape (N_samples, N_radii, N_per_center * N_radii, dim_feat)
        padded_samples = np.full(
            (N_samples, N_radii, N_per_center * N_radii, dim_feat),
            np.nan,
            dtype=np.float64
        )
        for k in range(N_radii):
            padded_samples[:, k, :(k+1)*N_per_center, :] = samples[:, :k+1, :, :].reshape(
                N_samples, (k+1)*N_per_center, dim_feat
            )
        return padded_samples
    else:
        return samples