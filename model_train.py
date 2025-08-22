# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:46:03 2025

@author: Admin
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')
