import numpy as np
import pandas as pd
from scipy.stats import entropy

def apply_noise(data: pd.DataFrame, mu: float = 0, sigma: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """
    Apply noise to the Pima Indians dataset features exactly as per the provided example logic.
    
    :param data: Original dataset (DataFrame)
    :param label_column: Name of the column containing the labels
    :param mu: Mean of the Gaussian noise (default 0)
    :param sigma: Standard deviation of the Gaussian noise (default 0.1)
    :param seed: Random seed for reproducibility
    :return: DataFrame with noise applied to features
    """
    np.random.seed(seed)
    
    # Create a copy of the data to avoid modifying the original
    noisy_data = data.copy()
    
    # 1. Apply noise to ordinal features (Pregnancies, Age) with range-based scaling
    ordinal_features = ['Pregnancies', 'Age']
    for feature in ordinal_features:
        max_value = data[feature].max()
        min_value = data[feature].min()
        feature_range = max_value - min_value
        
        # Apply Gaussian noise scaled by the feature's range and round
        noise = np.random.normal(mu, sigma * feature_range, size=noisy_data[feature].shape)
        noisy_data[feature] = noisy_data[feature] + noise
        noisy_data[feature] = noisy_data[feature].round()  # Round for ordinal features
        
        # Clip to ensure values stay within the original range
        noisy_data[feature] = noisy_data[feature].clip(lower=min_value, upper=max_value)

    # 2. Apply noise to continuous features proportionally to their range
    continuous_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    for feature in continuous_features:
        original_values = noisy_data[feature].values
        max_value = data[feature].max()
        min_value = data[feature].min()
        feature_range = max_value - min_value
        
        # Apply Gaussian noise, scaling sigma by the feature's range
        noise = np.random.normal(mu, sigma * feature_range, size=original_values.shape)
        noisy_data[feature] = original_values + noise
        
        # Clip values to be within the original range
        noisy_data[feature] = noisy_data[feature].clip(lower=min_value, upper=max_value)

    # Leave the label column unchanged
    return noisy_data

def flip_labels_general(data: pd.DataFrame, label_column: str, flip_prob: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """
    Flip the labels in the dataset with a given probability.
    
    :param data: Original dataset (DataFrame)
    :param label_column: Name of the column containing the labels
    :param flip_prob: Probability of flipping each label (default 0.1)
    :param seed: Random seed for reproducibility
    :return: DataFrame with labels flipped
    """
    np.random.seed(seed)
    
    # Create a copy of the data to avoid modifying the original
    flipped_data = data.copy()
    
    # Get unique labels in the label column and ensure they are integers
    possible_labels = flipped_data[label_column].unique().astype(int)
    
    def flip_label(label):
        if np.random.rand() < flip_prob:
            # Flip the label to any other possible label, ensuring integer output
            return int(np.random.choice([l for l in possible_labels if l != label]))
        return int(label)  # Ensure the original label remains an integer
    
    # Apply the flipping function to the label column
    flipped_data[label_column] = flipped_data[label_column].apply(flip_label)
    
    return flipped_data

def apply_data_dropping(data: pd.DataFrame, label_column: str, drop_prob: float = 0.1, seed: int = 42) -> pd.DataFrame:
    tmp_data = data.copy()
    return tmp_data.sample(frac=1-drop_prob, random_state=seed)

def compute_kl_divergence(original_data: pd.DataFrame, perturbed_data: pd.DataFrame, feature_columns: list) -> float:
    """
    Compute the Kullback-Leibler (KL) divergence between the original and perturbed datasets exactly as per the example.
    
    :param original_data: Original dataset (DataFrame)
    :param perturbed_data: Perturbed dataset (DataFrame)
    :param feature_columns: List of feature column names to include in the divergence calculation
    :return: KL divergence score
    """
    # Only use selected feature columns
    original_data = original_data[feature_columns]
    perturbed_data = perturbed_data[feature_columns]
    
    # Remove columns with identical values in perturbed data
    for feature in perturbed_data.columns:
        if len(perturbed_data[feature].unique()) == 1:
            original_data = original_data.drop(columns=[feature])
            perturbed_data = perturbed_data.drop(columns=[feature])
    
    # Convert to numpy arrays
    original_array = np.array(original_data)
    perturbed_array = np.array(perturbed_data)
    
    # Compute means
    mean_original = np.mean(original_array, axis=0)
    mean_perturbed = np.mean(perturbed_array, axis=0)
    
    # Compute covariances
    cov_original = np.cov(original_array, rowvar=False)
    cov_perturbed = np.cov(perturbed_array, rowvar=False)
    
    try:
        inv_cov_perturbed = np.linalg.inv(cov_perturbed)
    except np.linalg.LinAlgError:
        inv_cov_perturbed = np.linalg.pinv(cov_perturbed)  # Use pseudo-inverse if the matrix is singular
    
    # Calculate determinant values
    det_cov_original = np.linalg.det(cov_original)
    det_cov_perturbed = np.linalg.det(cov_perturbed)
    
    # Compute the KL divergence using the formula from the example
    kl_div = 0.5 * (
        np.log(det_cov_perturbed / det_cov_original) 
        - mean_original.shape[0] 
        + np.trace(inv_cov_perturbed @ cov_original)
        + (mean_original - mean_perturbed).T @ inv_cov_perturbed @ (mean_original - mean_perturbed)
    )
    
    # Handle potential NaN or very large values
    if np.isnan(kl_div) or abs(kl_div) > 1e7:
        kl_div = 10000.0
    elif kl_div < 0:
        kl_div = 0
    
    return kl_div