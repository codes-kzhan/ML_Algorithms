import numpy as np
from BayesianNetwork import BayesianNetwork


def complete_bin(number):
    bit = bin(number)[2:]
    number_of_zero = 12 - len(bit)
    return number_of_zero * '0' + bit


def compute_accuracy(pred, joint_probs):
    probs = np.array(map(lambda x: float(x), joint_probs[:,1]))
    return abs(pred - probs).sum()

     
joint_probs = [i.strip().split('\t') for i in open('joint.dat')]
data = np.array([complete_bin(int(i.strip())) for i in open('dataset.dat')])
joint_probs = np.array(map(lambda x: (complete_bin(int(x[0])), float(x[1])), joint_probs))

baseline_settings = {
    'IsSummer': [],
    'HasFlu': [],
    'HasFoodPoisoning': [],
    'HasHayFever': [],
    'HasPneumonia': [],
    'HasRespiratoryProblems': [],
    'HasGastricProblems': [],
    'HasRash': [],
    'Coughs': [],
    'IsFatigues': [],
    'Vomits': [],
    'HasFever': []
    }

settings = {
    'IsSummer': [],
    'HasFlu': ['IsSummer'],
    'HasFoodPoisoning': [],
    'HasHayFever': ['IsSummer'],
    'HasPneumonia': ['IsSummer'],
    'HasRespiratoryProblems': ['HasFlu', 'HasHayFever', 'HasPneumonia'],
    'HasGastricProblems': ['HasFoodPoisoning'],
    'HasRash': ['HasFlu', 'HasHayFever'],
    'Coughs': ['HasFlu', 'HasHayFever', 'HasPneumonia'],
    'IsFatigues': ['HasFlu', 'HasHayFever', 'HasPneumonia'],
    'Vomits': ['HasFlu', 'HasFoodPoisoning', 'HasGastricProblems'],
    'HasFever': ['HasFlu', 'HasPneumonia']
    }

settings2 = {
    'IsSummer': [],
    'HasFlu': ['IsSummer'],
    'HasFoodPoisoning': [],
    'HasHayFever': [],
    'HasPneumonia': ['IsSummer'],
    'HasRespiratoryProblems': ['HasFlu', 'HasHayFever', 'HasPneumonia', 'HasFoodPoisoning'],
    'HasGastricProblems': ['HasFlu', 'HasFoodPoisoning'],
    'HasRash': ['HasFoodPoisoning', 'HasHayFever'],
    'Coughs': ['HasFlu', 'HasPneumonia', 'HasRespiratoryProblems'],
    'IsFatigues': ['HasFlu', 'HasHayFever', 'HasPneumonia'],
    'Vomits': ['HasFoodPoisoning', 'HasGastricProblems'],
    'HasFever': ['HasFlu', 'HasPneumonia']
    }

feature_list = ['HasFever', 'Vomits', 'IsFatigues', 'Coughs', 'HasRash',
                'HasGastricProblems', 'HasRespiratoryProblems', 'HasPneumonia',
                'HasHayFever', 'HasFoodPoisoning', 'HasFlu', 'IsSummer']

model = BayesianNetwork(baseline_settings, feature_list)
model.fit(data)
prediction = model.predict(joint_probs)
score = compute_accuracy(prediction, joint_probs)
