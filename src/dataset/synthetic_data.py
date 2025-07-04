import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression, make_friedman1, make_friedman2, make_friedman3
from sklearn.datasets import make_spd_matrix

def apply_nonlinearity(X, y, regression_mode, n_informative):
    """Apply non-linearities to ALL informative features."""
    X_informative = X[:, :n_informative]  # First n_informative features are the true signals
    
    if regression_mode == "linear":
        return y
    
    elif regression_mode == "polynomial":
        y_nonlinear = y + 0.3 * np.sum(X_informative**2, axis=1) - 0.1 * np.sum(X_informative**3, axis=1)
    
    elif regression_mode == "interaction":
        interaction_terms = 0
        for i in range(n_informative):
            for j in range(i + 1, n_informative):
                interaction_terms += X_informative[:, i] * X_informative[:, j]
        y_nonlinear = y + 0.5 * interaction_terms
    
    elif regression_mode == "poly_interaction":
        quadratic_terms = 0.2 * np.sum(X_informative**2, axis=1)
        cubic_terms = -0.05 * np.sum(X_informative**3, axis=1)
        
        interaction_terms = 0
        for i in range(n_informative):
            for j in range(i + 1, n_informative):
                interaction_terms += X_informative[:, i] * X_informative[:, j]
        
        y_nonlinear = y + quadratic_terms + cubic_terms + 0.3 * interaction_terms
    
    elif regression_mode == "poly_cross":
        poly_terms = 0.4 * np.sum(X_informative[:, 1:]**2, axis=1)
        cross_terms = 0.7 * np.sum(X_informative[:, 0:1] * X_informative[:, 1:], axis=1)
        y_nonlinear = y + poly_terms + cross_terms
    
    elif regression_mode == "multiplicative_chain":
        # Multiplicative chain: x0 * x1 + x1 * x2 + x2 * x3 + ...
        chain_terms = 0
        for i in range(n_informative - 1):
            chain_terms += X_informative[:, i] * X_informative[:, i+1]
        y_nonlinear = y + 0.6 * chain_terms + 0.1 * np.sum(X_informative**2, axis=1)
    
    elif regression_mode == "rational":
        # Rational function terms (ratios of polynomials)
        numerator = 0.5 * X_informative[:, 0] + 0.3 * X_informative[:, 1]**2
        denominator = 1 + 0.2 * np.abs(X_informative[:, 2])
        y_nonlinear = y + numerator / (denominator + 1e-6)  # Small constant to avoid division by zero
    
    elif regression_mode == "exponential_interaction":
        interaction = 0
        for i in range(min(3, n_informative)):  # Use first 3 features for main interaction
            for j in range(i + 1, min(3, n_informative)):
                interaction += X_informative[:, i] * X_informative[:, j]
        y_nonlinear = y + 0.5 * np.exp(0.3 * interaction) + 0.2 * np.sum(X_informative, axis=1)
    
    elif regression_mode == "sigmoid_mix":
        # Sigmoid of linear combination plus polynomial terms
        weights = np.linspace(0.8, 1.2, n_informative)
        sigmoid_term = 2 / (1 + np.exp(-0.5 * np.dot(X_informative, weights)))
        poly_term = 0.1 * np.sum(X_informative**3, axis=1)
        y_nonlinear = y + sigmoid_term + poly_term
    
    elif regression_mode == "complex_hierarchy":
        # Complex hierarchical structure with polynomials and interactions
        main_effect = 0.4 * X_informative[:, 0]**2
        interaction_effect = 0.3 * X_informative[:, 0] * X_informative[:, 1]
        sub_interaction = 0.2 * X_informative[:, 2] * (X_informative[:, 3] + X_informative[:, 4])
        y_nonlinear = y + main_effect + interaction_effect + sub_interaction
    
    elif regression_mode == "piecewise":
        # Piecewise linear/non-linear effects
        term1 = np.where(X_informative[:, 0] > 0, 
                        0.5 * X_informative[:, 0]**2, 
                        -0.3 * X_informative[:, 0])
        term2 = 0.4 * np.sin(X_informative[:, 1]) * X_informative[:, 2]
        y_nonlinear = y + term1 + term2
    
    elif regression_mode == "hierarchical":
        # Complex hierarchical structure with nested dependencies
        # First level: base effects from first few features
        base_effect = 0.6 * X_informative[:, 0] + 0.4 * X_informative[:, 1]
        
        # Second level: interaction effects modulated by other features
        modulator = 0.5 + 0.3 * np.tanh(X_informative[:, 2])
        second_level = modulator * (X_informative[:, 3] * X_informative[:, 4])
        
        # Third level: higher-order interactions with remaining features
        third_level = 0
        for i in range(5, min(n_informative, 10)):
            weight = 0.2 / (i - 3)  # Diminishing contributions
            third_level += weight * X_informative[:, i] * base_effect
        
        y_nonlinear = y + base_effect + second_level + third_level
    
    elif regression_mode == "advanced_polynomial":
        # Advanced polynomial with varying degrees for different features
        degrees = np.array([2, 3, 1.5, 4, 2.5] + [2] * (n_informative - 5))[:n_informative]
        poly_terms = np.sum(np.sign(X_informative) * np.abs(X_informative) ** degrees[:, np.newaxis].T, axis=1)
        cross_terms = np.sum(X_informative[:, :-1] * X_informative[:, 1:], axis=1) if n_informative > 1 else 0
        y_nonlinear = y + 0.4 * poly_terms + 0.2 * cross_terms
    
    else:
        raise ValueError(f"Unknown regression_mode: {regression_mode}")
    
    return y_nonlinear

def get_setting_name_regression(regression_mode,
                                n_features,
                                n_informative, 
                                n_samples, 
                                noise,
                                bias,
                                tail_strength,
                                coef,
                                effective_rank,
                                random_seed):
    if 'friedman' in regression_mode:
        return regression_mode + f'_n_samples{n_samples}_noise{noise}_random_state{random_seed}'
    setting_name = (f'regression_{regression_mode}_n_feat{n_features}_n_informative{n_informative}_n_samples{n_samples}_'
                    f'noise{noise}_bias{bias}_random_state{random_seed}')
    if effective_rank is not None:
        setting_name += f'_effective_rank{effective_rank}_tail_strength{tail_strength}'
    return setting_name

def get_setting_name_classification(n_features,
                                    n_informative,
                                    n_redundant,
                                    n_repeated,
                                    n_classes,
                                    n_samples,
                                    n_clusters_per_class,
                                    class_sep,
                                    flip_y,
                                    random_seed,
                                    hypercube):
    setting_name = (f'n_feat{n_features}_n_informative{n_informative}_n_redundant{n_redundant}'
                    f'_n_repeated{n_repeated}_n_classes{n_classes}_n_samples{n_samples}'
                    f'_n_clusters_per_class{n_clusters_per_class}_class_sep{class_sep}'
                    f'_flip_y{flip_y}_random_state{random_seed}')
    if not hypercube:
        setting_name += f'_hypercube{hypercube}'
    return setting_name

def create_synthetic_classification_data_sklearn(n_features, 
                                   n_informative, 
                                   n_redundant, 
                                   n_repeated, 
                                   n_classes, 
                                   n_samples, 
                                   n_clusters_per_class, 
                                   class_sep, 
                                   flip_y, 
                                   hypercube,
                                   random_seed, 
                                   data_folder, 
                                   test_size=0.4, 
                                   val_size=0.1,
                                   force_create=False):
    
    setting_name = get_setting_name_classification(
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_samples=n_samples,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=flip_y,
        random_seed=random_seed,
        hypercube=hypercube
    )
    file_path = os.path.join(data_folder, f'{setting_name}.npz')
    
    if os.path.exists(file_path) and not force_create:
        data = np.load(file_path)
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
    else:
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features,
            n_informative=n_informative, 
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class, # Controls feature dependence by grouping samples into clusters
            class_sep=class_sep, # Higher values = features more separated between classes
            flip_y=flip_y, # Add noise by randomly flipping labels
            hypercube=hypercube,
            random_state=random_seed
        )
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_seed)
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        np.savez(file_path, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
    return setting_name, X_train, X_val, X_test, y_train, y_val, y_test


def create_synthetic_regression_data_sklearn(regression_mode,
                                             n_features, 
                                            n_informative, 
                                            n_samples, 
                                            noise, 
                                            bias, 
                                            random_seed, 
                                            data_folder, 
                                            test_size=0.4, 
                                            val_size=0.1,
                                            tail_strength=0.5,  # Only used if `effective_rank` is specified
                                            coef=False,         # Return true coefficients
                                            effective_rank=None  # Approximate rank of the data
                                           ):
    coef = False
    setting_name = get_setting_name_regression(
        regression_mode, 
        n_features=n_features,
        n_informative=n_informative,
        n_samples=n_samples,
        noise=noise,
        bias=bias,
        tail_strength=tail_strength,
        coef=coef,
        effective_rank=effective_rank,
        random_seed=random_seed
    )
    
    file_path = os.path.join(data_folder, f'{setting_name}.npz')
    if os.path.exists(file_path):
        data = np.load(file_path)
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        return setting_name, X_train, X_val, X_test, y_train, y_val, y_test
    elif regression_mode == "friedman1":
        X, y = make_friedman1(
            n_samples=n_samples,
            noise=noise,
            random_state=random_seed
        )
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_seed)
    elif regression_mode == "friedman2":
        X, y = make_friedman2(
            n_samples=n_samples,
            noise=noise,
            random_state=random_seed
        )
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_seed)
    elif regression_mode == "friedman3":
        X, y = make_friedman3(
            n_samples=n_samples,
            noise=noise,
            random_state=random_seed
        )
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_seed)
    else:
        out = make_regression(
                n_samples=n_samples, 
                n_features=n_features,
                n_informative=n_informative, 
                noise=noise,
                bias=bias,
                tail_strength=tail_strength,
                shuffle=False,
                coef=coef,
                effective_rank=effective_rank,
                random_state=random_seed
            )
    
        X, y =  out[0], out[1]
        
        y = apply_nonlinearity(X, y, regression_mode, n_informative)
        column_rng = np.random.RandomState(random_seed)
        col_indices = np.arange(X.shape[1])
        column_rng.shuffle(col_indices)
        X = X[:, col_indices]
        
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_seed)
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
    np.savez(file_path, 
                X_train=X_train, X_val=X_val, X_test=X_test, 
                y_train=y_train, y_val=y_val, y_test=y_test)
    return setting_name, X_train, X_val, X_test, y_train, y_val, y_test

def create_custom_synthetic_regression_data(regression_mode,
                                           n_features, 
                                           n_informative, 
                                           n_samples, 
                                           noise, 
                                           bias, 
                                           random_seed, 
                                           data_folder, 
                                           test_size=0.4, 
                                           val_size=0.1,
                                           tail_strength=0.5, 
                                           force_create = False, 
                                           effective_rank=None):
    """
    Create custom synthetic regression data manually without sklearn's make_regression.
    
    Args:
        regression_mode: Type of regression function to use
        n_features: Total number of features
        n_informative: Number of informative features
        n_samples: Total number of samples
        noise: Standard deviation of Gaussian noise
        bias: Constant bias term added to the target
        random_seed: Random seed for reproducibility
        data_folder: Directory to save the data
        test_size: Proportion of samples to use for testing
        val_size: Proportion of training samples to use for validation
        tail_strength: Parameter for controlling data distribution (not used in all modes)
        effective_rank: Parameter for controlling feature correlation (not used in all modes)
        
    Returns:
        tuple: (setting_name, X_train, X_val, X_test, y_train, y_val, y_test)
    """
    setting_name = get_setting_name_regression(
        regression_mode=regression_mode, 
        n_features=n_features,
        n_informative=n_informative,
        n_samples=n_samples,
        noise=noise,
        bias=bias,
        tail_strength=tail_strength,
        coef=False,
        effective_rank=effective_rank,
        random_seed=random_seed
    )
    
    file_path = os.path.join(data_folder, f'{setting_name}.npz')
    if "friedman" in regression_mode:
        if regression_mode == "friedman1":
            X, y = make_friedman1(
                n_samples=n_samples,
                noise=noise,
                random_state=random_seed
            )
            X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        elif regression_mode == "friedman2":
            X, y = make_friedman2(
                n_samples=n_samples,
                noise=noise,
                random_state=random_seed
            )
            X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        elif regression_mode == "friedman3":
            X, y = make_friedman3(
                n_samples=n_samples,
                noise=noise,
                random_state=random_seed
            )
            X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_seed)
        col_indices=None
    else:
        rng = np.random.RandomState(random_seed)
        X = rng.randn(n_samples, n_features)
        # feature_means = rng.normal(loc=0, scale=5, size=n_features) # You can adjust loc and scale
        # feature_std_devs = rng.uniform(low=0.5, high=3.0, size=n_features) # You can adjust low and high
        # X = X * feature_std_devs + feature_means

        if effective_rank is not None:
            # Create a low-rank covariance matrix to introduce correlations
            v = rng.uniform(0, 1, size=effective_rank)
            v = np.sort(v)[::-1]  # Sort decreasing
            v = v / np.sum(v) * effective_rank  # Normalize
            U = rng.randn(n_features, effective_rank)
            U, _ = np.linalg.qr(U)  # Orthogonalize
            cov = np.dot(U * v, U.T)
            cholesky = np.linalg.cholesky(cov + 1e-6 * np.eye(n_features))
            X = np.dot(X, cholesky)
        
        y = generate_target(X, regression_mode, n_informative, bias, tail_strength, rng)
        if noise > 0:
            y += rng.normal(0, noise, size=n_samples)

        col_indices = np.arange(n_features)
        print("Original column indices: ", col_indices)
        rng.shuffle(col_indices)
        print("Shuffled column indices: ", col_indices)
        X = X[:, col_indices]
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_seed)
        
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    print(file_path)
    np.savez(file_path, 
            X_train=X_train, X_val=X_val, X_test=X_test, 
            y_train=y_train, y_val=y_val, y_test=y_test)
    
    return setting_name, X_train, X_val, X_test, y_train, y_val, y_test, col_indices

def generate_target(X, regression_mode, n_informative, bias, tail_strength, rng):
    """
    Generate target values based on the regression mode.
    
    Args:
        X: Input features array of shape (n_samples, n_features)
        regression_mode: Type of regression function
        n_informative: Number of informative features
        bias: Constant bias term
        tail_strength: Parameter for controlling data distribution
        rng: Random number generator
        
    Returns:
        np.ndarray: Target values
    """
    n_samples, n_features = X.shape
    
    X_informative = X[:, :n_informative]
    
    if regression_mode == "linear":
        coeffs = rng.uniform(-1, 1, size=n_informative)
        y = np.dot(X_informative, coeffs)
        
    elif regression_mode == "polynomial":
        degrees = np.arange(1, min(5, n_informative) + 1)
        terms = [np.power(X_informative[:, i % n_informative], degrees[i % len(degrees)]) 
                for i in range(n_informative)]
        coeffs = rng.uniform(-1, 1, size=len(terms))
        y = np.sum([c * term for c, term in zip(coeffs, terms)], axis=0)
        
    elif regression_mode == "trigonometric":
        freqs = rng.uniform(0.1, 2.0, size=n_informative)
        phases = rng.uniform(0, 2*np.pi, size=n_informative)
        amplitudes = rng.uniform(0.5, 2.0, size=n_informative)
        
        sin_terms = [amplitudes[i] * np.sin(freqs[i] * X_informative[:, i] + phases[i]) 
                    for i in range(n_informative)]
        cos_terms = [amplitudes[i] * np.cos(freqs[i] * X_informative[:, i % n_informative] + phases[i]) 
                    for i in range(min(3, n_informative))]
        
        y = np.sum(sin_terms + cos_terms, axis=0)
        
    elif regression_mode == "exponential":
        bases = rng.uniform(1.1, 1.5, size=min(3, n_informative))
        coeffs = rng.uniform(-0.5, 0.5, size=min(3, n_informative))
        
        exp_terms = [coeffs[i] * np.exp(bases[i] * X_informative[:, i]) 
                    for i in range(min(3, n_informative))]
        linear_terms = [0.2 * X_informative[:, i] for i in range(3, n_informative)]
        
        y = np.sum(exp_terms + linear_terms, axis=0)
        
    elif regression_mode == "logistic":
        slopes = rng.uniform(1.0, 3.0, size=min(5, n_informative))
        midpoints = rng.uniform(-0.5, 0.5, size=min(5, n_informative))
        
        sigmoid_terms = [1.0 / (1.0 + np.exp(-slopes[i] * (X_informative[:, i] - midpoints[i])))
                        for i in range(min(5, n_informative))]
        linear_terms = [0.1 * X_informative[:, i] for i in range(5, n_informative)]
        
        y = np.sum(sigmoid_terms + linear_terms, axis=0)
        
    elif regression_mode == "periodic":
        main_freq = 2.0 * np.pi * rng.uniform(1.5, 2.5)
        primary_signal = np.sin(main_freq * X_informative[:, 0])
        if n_informative > 1:
            mod_signal = 0.5 + 0.5 * np.tanh(2 * X_informative[:, 1])
            primary_signal *= mod_signal
        if n_informative > 2:
            freq_mod = 0.2 * X_informative[:, 2]
            secondary_signal = 0.3 * np.sin(main_freq * (1 + freq_mod) * X_informative[:, 0])
            primary_signal += secondary_signal
        additional_terms = [0.1 * np.sin(rng.uniform(1, 5) * X_informative[:, i]) 
                           for i in range(3, n_informative)]
        y = primary_signal + np.sum(additional_terms, axis=0) if additional_terms else primary_signal
    elif regression_mode == "piecewise_linear":
        y = np.zeros(n_samples)
        for i in range(n_informative):
            feature = X_informative[:, i]
            t1 = rng.uniform(-0.5, 0.5)
            t2 = rng.uniform(0.5, 1.5)
            slope1 = rng.uniform(0.5, 1.5)
            slope2 = rng.uniform(-1.5, -0.5)
            slope3 = rng.uniform(0.1, 1.0)
            # Piecewise function:
            # If x < t1:      y = slope1 * x
            # If t1 <= x < t2:y = slope2 * x + offset
            # If x >= t2:     y = slope3 * x + offset2
            segment1 = feature < t1
            segment2 = (feature >= t1) & (feature < t2)
            segment3 = feature >= t2
            
            offset = rng.uniform(-0.5, 0.5)
            offset2 = rng.uniform(-1.0, 1.0)

            y += (
                segment1 * (slope1 * feature) +
                segment2 * (slope2 * feature + offset) +
                segment3 * (slope3 * feature + offset2)
            )
    else:
        y = np.zeros(n_samples)
        y = apply_nonlinearity(X, y, regression_mode, n_informative)
        return y
    
    # Add bias term
    return y + bias