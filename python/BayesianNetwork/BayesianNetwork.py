import numpy as np
from itertools import combinations


class BayesianNetwork(object):

    def __init__(self, settings, feature_list):
        self.settings = settings
        self.feature_list = feature_list
        self.counts, self.parent_counts = self.initial_counts()
        self.conditional_probs = {}
        self.parent_probs = {}

    def initial_counts(self):
        """
        Initialize the counting tables for conditional probabilities
        """
        settings = self.settings
        counts = {}
        parent_counts = {}
        
        for key in settings:
            parents = sorted(settings[key])
            if not parents:
                counts[key] = [0, 0]
            else:
                counts[key] = {}
                associations = [p+v for p in parents for v in ['0', '1']]
                combs = list(combinations(associations, len(parents)))
                # Eliminate combinations that have duplicate variables
                tables = [''.join(k) for k in combs if sorted([h[:-1] for h in k])==sorted(list(set([h[:-1] for h in k])))]
                for tab in tables:
                    counts[key][tab] = [0, 0]
                    parent_counts[tab] = 0
                    
        return counts, parent_counts

    def build_prob_table(self, size):
        """
        Build conditional probability tables using counting tables
        """
        settings = self.settings
        counts = self.counts
        parent_counts = self.parent_counts
        conditional_probs = {}
        parent_probs = {}
        
        for key in settings:
            parents = sorted(settings[key])
            if not parents:
                cnt = counts[key]
                conditional_probs[key] = [float(cnt[0])/size, float(cnt[1])/size]
            else:
                conditional_probs[key] = {}
                associations = [p+v for p in parents for v in ['0', '1']]
                combs = list(combinations(associations, len(parents)))
                # Eliminate combinations that have duplicate variables
                tables = [''.join(k) for k in combs if sorted([h[:-1] for h in k])==sorted(list(set([h[:-1] for h in k])))]
                for tab in tables:
                    cnt = counts[key][tab]
                    parent_cnt = parent_counts[tab]
                    conditional_probs[key][tab] = [float(cnt[0])/size, float(cnt[1])/size]
                    parent_probs[tab] = float(parent_cnt)/size

        return conditional_probs, parent_probs

    def fit(self, data):
        """
        Fit the training data and build conditional probability tables
        """
        settings = self.settings
        feature_list = self.feature_list
        counts = self.counts
        parent_counts = self.parent_counts
        
        for bit in data:
            tab_list = []
            for t, i in enumerate(bit):
                i = int(i)
                feature = feature_list[t]
                parents = settings[feature]
                if not parents:
                    counts[feature][i] += 1
                else:
                    tab = ''
                    for par in sorted(parents):
                        index = feature_list.index(par)
                        tab += par
                        tab += str(bit[index])
                    counts[feature][tab][i] += 1
                    if tab not in tab_list:
                        tab_list.append(tab)
                        parent_counts[tab] += 1
                        
        self.conditional_probs, self.parent_probs = self.build_prob_table(len(data))

    def compute_prob(self, bit):
        """
        Predict the probability for given bit(feature values)
        """
        feature_list = self.feature_list
        settings = self.settings
        conditional_probs = self.conditional_probs
        parent_probs = self.parent_probs
        prob = 1.0
        
        for t, i in enumerate(bit):
            i = int(i)
            feature = feature_list[t]
            parents = settings[feature]
            if not parents:
                prob *= conditional_probs[feature][i]
            else:
                tab = ''
                for par in sorted(parents):
                    index = feature_list.index(par)
                    tab += par
                    tab += str(bit[index])
                prob *= (conditional_probs[feature][tab][i]/parent_probs[tab])

        return prob

    def predict(self, joint_probs):
        """
        Make predictions for every bits(feature values) in
        the joint probability list
        """
        prediction = []
        for bit, prob in joint_probs:
            prediction.append(self.compute_prob(bit))
        return np.array(prediction)
