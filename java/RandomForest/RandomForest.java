package RandomForest;

import java.io.IOException;
import java.util.*;

import static RandomForest.DecisionTree.*;
import static RandomForest.util.*;

public class RandomForest {
	
	int n_estimators = 100;
	int max_depth = 999;
	int max_features = 10;
	String criteria = "None";
	List<DecisionTree> forest = new ArrayList<DecisionTree>();
	List<String> targetList = new ArrayList<String>();
	
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
		Set<String> targetSet = new HashSet<String>(y);
		targetList = new ArrayList<String>(targetSet);
		Collections.sort(targetList);
		
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
	
	public List<String> predict(List<double[]> data) {
		List<String> ret = new ArrayList<String>();
		List<double[]> retProba = predictProba(data);
		
		for (double[] tmp : retProba) {
			int index = argmax(tmp);
			ret.add(targetList.get(index));
		}
		
		return ret;
	}
	
	public double score(List<double[]> data, List<String> y) {
		List<String> yPred = new ArrayList<String>();
		yPred = predict(data);
		return correctNum(y, yPred) / ( (double) y.size() );
	}
	
	public double logLoss(List<double[]> data, List<String> y) {
		List<double[]> yPred = new ArrayList<double[]>();
		yPred = predictProba(data);
		return multiLogLoss(yPred, y, 1E-15);
	}

	public static void main(String[] args) throws IOException {
		String filename = "/home/yejiming/desktop/python/ML_Algorithms/dataset.csv";
		List ret = loadData(filename);
		List<double[]> data = (List<double[]>) ret.get(0);
		List<String> y = (List<String>) ret.get(1);
		String testfile = "/home/yejiming/desktop/Kaggle/OttoGroup/test.csv";
		List ret2 = loadTest(testfile);
		List<double[]> dataTest = (List<double[]>) ret2.get(0);
		List<String> ids = (List<String>) ret2.get(1);
		
		RandomForest rf = new RandomForest();
		rf.train(data, y);
		List<double[]> yPred = rf.predictProba(dataTest);
		String submission = "/home/yejiming/desktop/Kaggle/OttoGroup/rf_submission.csv";
		submit(yPred, ids, submission);
	}
}
