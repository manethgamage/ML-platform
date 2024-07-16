import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE

def check_imbalance(y):
    counter = Counter(y)
    majority_class = max(counter.values())
    minority_class = min(counter.values())
    imbalance_ratio = majority_class / minority_class
    return imbalance_ratio

def apply_oversampling(x, y, method='auto', imbalance_threshold=1.5):
    imbalance_ratio = check_imbalance(y)

    if imbalance_ratio > imbalance_threshold:
        if method == 'auto':
            if imbalance_ratio > 2:
                method = 'smote'
            else:
                method = 'random'

        if method == 'random':
            oversampler = RandomOverSampler(random_state=42)
        elif method == 'smote':
            oversampler = SMOTE(random_state=42)

        X_resampled, y_resampled = oversampler.fit_resample(x, y)
        return X_resampled, y_resampled
    else:
        return x, y
    