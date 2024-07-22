import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

def check_imbalance(y):
    counter = Counter(y)
    majority_class = max(counter.values())
    minority_class = min(counter.values())
    imbalance_ratio = majority_class / minority_class
    return imbalance_ratio , minority_class

def apply_oversampling(x, y, method='auto', imbalance_threshold=1.5, minority_threshold = 100000):
    imbalance_ratio, minority_class = check_imbalance(y)
    if minority_class > minority_threshold:
        print("Applying Random UnderSampling for handling class imbalance...")
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(x, y)
        return X_resampled , y_resampled
    else:
        if imbalance_ratio > imbalance_threshold:
            if method == 'auto':
                if imbalance_ratio > 2:
                    method = 'smote'
                else:
                    method = 'random'

            if method == 'random':
                oversampler = RandomOverSampler(random_state=42)
            elif method == 'smote':
                print('aplying smote')
                oversampler = SMOTE(random_state=42)

            X_resampled, y_resampled = oversampler.fit_resample(x, y)
            return X_resampled, y_resampled
        else:
            return x, y
    