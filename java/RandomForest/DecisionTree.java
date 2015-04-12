package RandomForest;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

import static RandomForest.util.*;


class TreeNode {
	int feature;
	double value;
	String maxClass;
	double[] probas;
	TreeNode left = null;
	TreeNode right = null;
	
	public TreeNode(int feature, double value) {
		this.feature = feature;
		this.value = value;
	}
	
	public TreeNode(int feature, double value, String maxClass, double[] probas, TreeNode left, TreeNode right) {
		this.feature = feature;
		this.value = value;
		this.maxClass = maxClass;
		this.probas = probas;
		this.left = left;
		this.right = right;
	}
}


class BTree {
	
	TreeNode root;
	
	public BTree(TreeNode root) {
		this.root = root;
	}
	
	public BTree() {}

	public void preOrder(TreeNode node) {
		
		if (node.left == null) {
			System.out.print(node.feature);
			System.out.print("   ");
			System.out.print(node.value);
			System.out.print("   ");
			System.out.println(node.maxClass);
		}
		
		else {
			System.out.print(node.feature);
			System.out.print("   ");
			System.out.print(node.value);
			System.out.print("   ");
			System.out.println(node.maxClass);
			preOrder(node.left);
			preOrder(node.right);
		}
		
	}
	
	public int getNumLeafs(TreeNode node) {
		int numLeafs = 0;
		
		if (node.left == null) {
			numLeafs += 1;
		}
		
		else {
			numLeafs += getNumLeafs(node.left);
			numLeafs += getNumLeafs(node.right);
		}
		
		return numLeafs;
	}
	
	public String getClass(double[] x, TreeNode node) {
		
		if (node.left == null) {
			return node.maxClass;
		}
		
		else {
			if (x[node.feature] > node.value)
				return getClass(x, node.right);
			else
				return getClass(x, node.left);
		}
		
	}
	
	public double[] getProba(double[] x, TreeNode node) {
		
		if (node.left == null) {
			return node.probas;
		}
		
		else {
			if (x[node.feature] > node.value)
				return getProba(x, node.right);
			else
				return getProba(x, node.left);
		}
		
	}
}


public class DecisionTree {
	
	double tol = 0.0001;
	int leastSample = 1;
	int maxDepth = 999;
	String merge = "pep";
	int maxFeatures = 10;
	List<String> targetList = new ArrayList<String>();
	BTree tree = new BTree();
	
	public DecisionTree(double tol, int leastSample, int maxDepth, String merge, int maxFeatures) {
		this.tol = tol;
		this.leastSample = leastSample;
		this.maxDepth = maxDepth;
		this.merge = merge;
		this.maxFeatures = maxFeatures;
	}
	
	public DecisionTree() {
	}
	
	public List binSplitDataSet(List<double[]> data, List<String> y, int feature, double value) {
		List ret = new ArrayList();
		List<double[]> mat0 = new ArrayList<double[]>();
		List<double[]> mat1 = new ArrayList<double[]>();
		List<String> y0 = new ArrayList<String>();
		List<String> y1 = new ArrayList<String>();
		
		for (int i = 0; i < data.size(); i++) {
			double[] row = data.get(i);
			if (row[feature] <= value) {
				mat0.add(row);
				y0.add(y.get(i));
			}
			else {
				mat1.add(row);
				y1.add(y.get(i));
			}
		}
		
		ret.add(mat0); ret.add(y0); ret.add(mat1); ret.add(y1);
		return ret;
	}
	
	public List chooseBestSplit(List<double[]> data, List<String> y) {
		List ret = new ArrayList();
		Set<String> ySet = new HashSet<String>(y);
		
		if (ySet.size() == 1) {
			ret.add(null); ret.add(null); ret.add(y.get(0)); ret.add(probaArray(y));
			return ret;
		}
		
		int n = data.get(0).length;
		double S = gini(y);
		double bestS = 1.0;
		int bestIndex = 0;
		double bestValue = 0.0;
		List<Integer> featureList = new ArrayList<Integer>();
		int visitedFeatures = 0;
		
		for (int i = 0; i < n; i++)
			featureList.add(i);
		
		while (visitedFeatures < maxFeatures && featureList.size() > 0) {
			Random r = new Random();
			int randNum = r.nextInt(featureList.size());
			int featIndex = featureList.get(randNum);
			featureList.remove(randNum);
			List<Double> column = new ArrayList<Double>();
			
			for (double[] i : data) 
				column.add(i[featIndex]);
			
			Set<Double> featSet = new HashSet<Double>(column);
			if (featSet.size() == 1)
				continue;
			
			visitedFeatures += 1;
			/*double lastS = (double) 1.0;*/
			
			for (double splitVal : featSet) {
				List splitted = binSplitDataSet(data, y, featIndex, splitVal);
				List<double[]> mat0 = (ArrayList<double[]>) splitted.get(0);
				List<String> y0 = (ArrayList<String>) splitted.get(1);
				List<double[]> mat1 = (ArrayList<double[]>) splitted.get(2);
				List<String> y1 = (ArrayList<String>) splitted.get(3);
				
				if (mat0.size() < leastSample || mat1.size() < leastSample)
					continue;
				
				double r0 = ((double) y0.size()) / y.size();
				double r1 = ((double) y1.size()) / y.size();
				double newS = r0 * gini(y0) + r1 * gini(y1);
				
				/*if (newS > lastS)
					break;*/
				
				if (newS < bestS) {
					bestIndex = featIndex;
					bestValue = splitVal;
					bestS = newS;
				}
				
				/*lastS = newS;*/
			}
		}
		
		if (S - bestS < tol) {
			ret.add(null); ret.add(null); ret.add(classify(y)); ret.add(probaArray(y));
			return ret;
		}
		
		List splitted = binSplitDataSet(data, y, bestIndex, bestValue);
		List<double[]> mat0 = (ArrayList<double[]>) splitted.get(0);
		List<String> y0 = (ArrayList<String>) splitted.get(1);
		List<double[]> mat1 = (ArrayList<double[]>) splitted.get(2);
		List<String> y1 = (ArrayList<String>) splitted.get(3);
		
		if (mat0.size() < leastSample || mat1.size() < leastSample) {
			ret.add(null); ret.add(null); ret.add(classify(y)); ret.add(probaArray(y));
			return ret;
		}
		
		ret.add(bestIndex); ret.add(bestValue); ret.add(classify(y)); ret.add(probaArray(y));
		return ret;
	}
	
	public TreeNode createTree(List<double[]> data, List<String> y, int depth) {
		TreeNode retTree = new TreeNode(-1, -1);
		
		if (depth > maxDepth) {
			retTree.maxClass = classify(y);
			retTree.probas = probaArray(y);
			return retTree;
		}
		
		List bestSplit = chooseBestSplit(data, y);
		Object feat = bestSplit.get(0); Object val = bestSplit.get(1); String mc = (String) bestSplit.get(2); double[] probas = (double[]) bestSplit.get(3);
		if (feat == null) {
			retTree.maxClass = mc;
			retTree.probas = probas;
			return retTree;
		}
		
		retTree.feature = (int) feat;
		retTree.value = (double) val;
		retTree.maxClass = mc;
		retTree.probas = probas;
		List splitted = binSplitDataSet(data, y,  (int) feat, (double) val);
		List<double[]> left = (ArrayList<double[]>) splitted.get(0);
		List<String> yleft = (ArrayList<String>) splitted.get(1);
		List<double[]> right = (ArrayList<double[]>) splitted.get(2);
		List<String> yright = (ArrayList<String>) splitted.get(3);
		retTree.left = createTree(left, yleft, depth + 1);
		retTree.right = createTree(right, yright, depth + 1);
		
		return retTree;
	}
	
	public boolean isLeaf(TreeNode node) {
		return (node.left == null);
	}
	
	private TreeNode prune_pep(TreeNode node, List<double[]> data, List<String> y) {
		if (isLeaf(node))
			return node;
		
		int numLeafs = tree.getNumLeafs(node);
		long length = y.size();
		double errorNoMerge = (1 - score(data, y)) * length + 0.5 * numLeafs;
		double varNoMerge = Math.sqrt(errorNoMerge * (1 - errorNoMerge / length));
		List<String> yPred = new ArrayList<String>();
		
		for (int i = 0; i < length; i++) {
			yPred.add(node.maxClass);
		}
		
		double errorMerge = length - correctNum(y, yPred) + 0.5;
		
		if (errorMerge < errorNoMerge + varNoMerge) {
			node.feature = -1; node.value = -1; node.left = null; node.right = null;
			return node;
		}
		
		else {
			List splitted = binSplitDataSet(data, y, node.feature, node.value);
			List<double[]> left = (ArrayList<double[]>) splitted.get(0);
			List<String> yleft = (ArrayList<String>) splitted.get(1);
			List<double[]> right = (ArrayList<double[]>) splitted.get(2);
			List<String> yright = (ArrayList<String>) splitted.get(3);
			
			if (!isLeaf(node.left))
				node.left = prune_pep(node.left, left, yleft);
			
			if (!isLeaf(node.right))
				node.right = prune_pep(node.right, right, yright);
			
			return node;
		}
	}
	
	public void train(List<double[]> data, List<String> y) {
		Set<String> targetSet = new HashSet<String>(y);
		targetList = new ArrayList<String>(targetSet);
		Collections.sort(targetList);
		
		long length = y.size();
		long cutPoint = length * 9 / 10;
		List<String> ytrain = new ArrayList<String>();
		List<double[]> train = new ArrayList<double[]>();
		List<String> ytest = new ArrayList<String>();
		List<double[]> test = new ArrayList<double[]>();
		
		for (int i = 0; i < data.size(); i ++) {
			if (i < cutPoint) {
				train.add(data.get(i));
				ytrain.add(y.get(i));
			}
			
			else {
				test.add(data.get(i));
				ytest.add(y.get(i));
			}
		}
		
		if (merge == "None") {
			TreeNode root = createTree(data, y, 0);
			tree = new BTree(root);
		}
		
		else if (merge == "pep" ) {
			TreeNode root = createTree(data, y, 0);
			tree = new BTree(root);
			root = prune_pep(root, data, y);
			tree = new BTree(root);
		}
		
		else
			System.out.println("Prune type does not exist!");
	}
	
	public List predict(List<double[]> data) {
		List<String> ret = new ArrayList<String>();
		
		for (double[] x : data) {
			String pred = tree.getClass(x, tree.root);
			ret.add(pred);
		}
		
		return ret;
	}
	
	public double[] probaArray(List<String> y) {
		DecimalFormat df = new DecimalFormat( "0.00000 ");
		double[] ret = new double[targetList.size()];
		
		for (int i = 0; i < ret.length; i ++) 
			ret[i] = (double) 0.0;
		
		for (String item : y) {
			int index = targetList.indexOf(item);
			ret[index] += 1;
		}
		
		for (int i = 0; i < ret.length; i ++) {
			ret[i] /= y.size();
			ret[i] = Double.parseDouble(df.format(ret[i]));
		}
		
		return ret;
	}
	
	public List<double[]> predictProba(List<double[]> data) {
		List<double[]> ret = new ArrayList<double[]>();
		
		for (double[] x : data) {
			double[] pred = tree.getProba(x, tree.root).clone();
			ret.add(pred);
		}
		
		return ret;
	}
	
	public double score(List<double[]> data, List<String> y) {
		List<String> yPred = new ArrayList<String>();
		yPred = predict(data);
		return correctNum(y, yPred) / ( (double) y.size() );
	}
	
	public static void main(String[] args) throws IOException  {
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
		
		DecisionTree learner = new DecisionTree(0.0001, 1, 999, "pep", 10);
		learner.train(train, ytrain);
		List<double[]> yp = learner.predictProba(test);
		for (int i = 0; i < yp.size(); i++) {
			double[] p1 = yp.get(i);
			for (int j = 0; j < p1.length; j++) {
				p1[j] += 1;
			}
		}
		double[] p = yp.get(0);
		for (double i : p)
			System.out.println(i);
	}
}
