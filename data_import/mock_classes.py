# File: klm87728-anemos/data_import/mock_classes.py
from scipy import stats
import numpy as np
class MockDistFit:
    def __init__(self, name='norm', loc=0, scale=1):
        self.model = {
            'name': name,
            'distr': stats.norm, # Using scipy.stats.norm as base
            'arg': (),
            'loc': loc,
            'scale': scale
        }


class MockDisruptions:
    def __init__(self):
        # Existing disruption levels setup
        self.disruption_levels = [
            {'levelId': 0, 'level': 'norm', 'delays_fitted_dist': MockDistFit(loc=5, scale=5),
             'events_fitted_dist': MockDistFit(loc=1, scale=0), 'probability_no_delay': 0.8, 'delays_duration': [],
             'events_duration': []},
            {'levelId': 1, 'level': 'low', 'delays_fitted_dist': MockDistFit(loc=30, scale=10),
             'events_fitted_dist': MockDistFit(loc=2, scale=1), 'probability_no_delay': 0.5, 'delays_duration': [],
             'events_duration': []},
            {'levelId': 2, 'level': 'high', 'delays_fitted_dist': MockDistFit(loc=60, scale=20),
             'events_fitted_dist': MockDistFit(loc=4, scale=2), 'probability_no_delay': 0.2, 'delays_duration': [],
             'events_duration': []}
        ]

        # FIX: Rename to transition_probability_matrix and convert to a numpy array
        self.transition_probability_matrix = np.array([
            [0.8, 0.1, 0.1],  # From Norm to: Norm, Low, High
            [0.2, 0.7, 0.1],  # From Low to: Norm, Low, High
            [0.1, 0.3, 0.6]  # From High to: Norm, Low, High
        ])