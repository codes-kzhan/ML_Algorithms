package RandomForest;

import java.io.IOException;
import java.util.*;

import static RandomForest.DecisionTree.*;
import static RandomForest.util.*;

public class RandomForest {
	
	int n_estimators = 10;
	int max_depth = 999;
	int max_features = 10;
	String criteria = "None";
	List<DecisionTree> forest = new ArrayList<DecisionTree>();
	
	public RandomForest() {
		for (int i = 0; i < n_estimators; i++) {
			DecisionTree tree = new DecisionTree(0.0001, 1, max_depth, "pep", max_features);
			forest.add(tree);
		}
	}
	
	public RandomForest(int n_estimators, int max_depth, int max_features, String criteria) {
		this.n_estimators = n_estimators;
		this.max_depth = max_depth;
		this.max_features = max_features;
		this.criteria = criteria;
		
		for (int i = 0; i < n_estimators; i++) {
			DecisionTree tree = new DecisionTree(0.0001, 1, max_depth, "pep", max_features);
			forest.add(tree);
		}
	}
	
	public void train(List<double[]> data, List<String> y) {
		
		for (int i = 0; i < n_estimators; i++) {
			List retList = bootstrap(data, y);
			List<double[]> dataBoot = new ArrayList<double[]>();
			List<String> yBoot = new ArrayList<String>();
			dataBoot = (List<double[]>) retList.get(0);
			yBoot = (List<String>) retList.get(1);
			System.out.println(i + 1);
			forest.get(i).train(dataBoot, yBoot);
		}
	}
	
	public List<double[]> predictProba(List<double[]> data) {
		List<double[]> ret = forest.get(0).predictProba(data);
		
		for (int i = 1; i < forest.size(); i++) {
			DecisionTree currTree = forest.get(i);
			List<double[]> probas = currTree.predictProba(data);
			for (int j = 0; j < probas.size(); j++) {
				double[] p = probas.get(j);
				for (int k = 0; k < p.length; k++)  
					ret.get(j)[k] += p[k];
			}
		}
		
		for (int j = 0; j < ret.size(); j++) {
			double[] p = ret.get(j);
			for (int k = 0; k < p.length; k++) {
				p[k] /= forest.size();
			}
		}
		
		return ret;
	}

	public static void main(String[] args) throws IOException {
		String filename = "/home/yejiming/desktop/python/ML_Algorithms/dataset.csv";
		List ret = loadData(filename);
		List<double[]> data = (List<double[]>) ret.get(0);
		List<String> y = (List<String>) ret.get(1);
		List<String> ytrain = new ArrayList<String>();
		List<double[]> train = new ArrayList<double[]>();
		List<String> ytest = new ArrayList<String>();
		List<double[]> test = new ArrayList<double[]>();
		
		for (int i = 0; i < data.size(); i ++) {
			if (i < 50000) {
				train.add(data.get(i));
				ytrain.add(y.get(i));
			}
			
			else {
				test.add(data.get(i));
				ytest.add(y.get(i));
			}
		}
		
		RandomForest rf = new RandomForest();
		rf.train(train, ytrain);
		List<double[]> yp = rf.predictProba(test);
		double[] p = yp.get(9999);
		for (double i : p)
			System.out.println(i);
	}
}
