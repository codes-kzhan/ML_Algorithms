package RandomForest2;

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
	String merge = "None";
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
	
	public List binSplitDataSet(List<DataRow> dataSet, int feature, double value) {
		List ret = new ArrayList();
		List<DataRow> left = new ArrayList<DataRow>();
		List<DataRow> right = new ArrayList<DataRow>();
		
		int size = dataSet.size();
		for (int i = 0; i < size; i++) {
			DataRow row = dataSet.get(i);
			double[] x = row.x;
			if (x[feature] <= value) {
				left.add(row);
			}
			else {
				right.add(row);
			}
		}
		
		ret.add(left); ret.add(right);
		return ret;
	}
	
	public List chooseBestSplit(List<DataRow> dataSet) {
		List ret = new ArrayList();
		int n = dataSet.get(0).x.length;
		int size = dataSet.size();
		
		double S = gini(dataSet);
		if (S == 1.0) {
			ret.add(null); ret.add(null); ret.add(dataSet.get(0).y); ret.add(probaArray(dataSet));
			return ret;
		}
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
			for (DataRow item : dataSet) { 
				double[] x = item.x;
				column.add(x[featIndex]);
			}
			
			Set<Double> featSet = new HashSet<Double>(column);
			if (featSet.size() == 1)
				continue;
			
			visitedFeatures += 1;
			List<Double> featList = new ArrayList<Double>(featSet);
			Collections.sort(featList);
			Collections.sort(dataSet,new Comparator<DataRow>() {
	            public int compare(DataRow arg0, DataRow arg1) {
	                return ((Double) arg0.x[featIndex]).compareTo((Double) arg1.x[featIndex]);
	            }
	        });
			
			List<Double> column2 = new ArrayList<Double>();
			for (DataRow item : dataSet) { 
				double[] x = item.x;
				column2.add(x[featIndex]);
			}
			int lastIndex = 0;
			
			/*double lastS = (double) 1.0;*/
			
			for (int i = 1; i < featList.size(); i++) {
				double splitVal = featList.get(i);
				int index = column2.indexOf(splitVal);
				
				if (index - lastIndex < leastSample || size - index < leastSample)
					continue;
				
				List<DataRow> ds0 = dataSet.subList(0, index);
				List<DataRow> ds1 = dataSet.subList(index, size - 1);
				double r0 = ((double) ds0.size()) / size;
				double r1 = ((double) ds1.size()) / size;
				double newS = r0 * gini(ds0) + r1 * gini(ds1);
				
				/*if (newS > lastS)
					break;*/
				
				if (newS < bestS) {
					bestIndex = featIndex;
					bestValue = splitVal;
					bestS = newS;
					lastIndex = index;
				}
				
				/*lastS = newS;*/
			}
		}
		
		if (S - bestS < tol) {
			ret.add(null); ret.add(null); ret.add(classify(dataSet)); ret.add(probaArray(dataSet));
			return ret;
		}
		
		List splitted = binSplitDataSet(dataSet, bestIndex, bestValue);
		List<DataRow> ds0 = (List<DataRow>) splitted.get(0);
		List<DataRow> ds1 = (List<DataRow>) splitted.get(1);
		
		if (ds0.size() < leastSample || ds1.size() < leastSample) {
			ret.add(null); ret.add(null); ret.add(classify(dataSet)); ret.add(probaArray(dataSet));
			return ret;
		}
		
		ret.add(bestIndex); ret.add(bestValue); ret.add(classify(dataSet)); ret.add(probaArray(dataSet));
		return ret;
	}
	
	public TreeNode createTree(List<DataRow> dataSet, int depth) {
		TreeNode retTree = new TreeNode(-1, -1);
		
		if (depth > maxDepth) {
			retTree.maxClass = classify(dataSet);
			retTree.probas = probaArray(dataSet);
			return retTree;
		}
		
		List bestSplit = chooseBestSplit(dataSet);
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
		List splitted = binSplitDataSet(dataSet,  (int) feat, (double) val);
		List<DataRow> left = (List<DataRow>) splitted.get(0);
		List<DataRow> right = (List<DataRow>) splitted.get(1);
		retTree.left = createTree(left, depth + 1);
		retTree.right = createTree(right, depth + 1);
		
		return retTree;
	}
	
	public boolean isLeaf(TreeNode node) {
		return (node.left == null);
	}
	
	private TreeNode prune_pep(TreeNode node, List<DataRow> dataSet) {
		if (isLeaf(node))
			return node;
		
		int numLeafs = tree.getNumLeafs(node);
		long size = dataSet.size();
		List inversed = inverse(dataSet);
		List<double[]> data = (List<double[]>) inversed.get(0);
		List<String> y = (List<String>) inversed.get(1);
		double errorNoMerge = (1 - score(data, y)) * size + 0.5 * numLeafs;
		double varNoMerge = Math.sqrt(errorNoMerge * (1 - errorNoMerge / size));
		List<String> yPred = new ArrayList<String>();
		
		for (int i = 0; i < size; i++) {
			yPred.add(node.maxClass);
		}
		
		double errorMerge = size - correctNum(y, yPred) + 0.5;
		
		if (errorMerge < errorNoMerge + varNoMerge) {
			node.feature = -1; node.value = -1; node.left = null; node.right = null;
			return node;
		}
		
		else {
			List splitted = binSplitDataSet(dataSet, node.feature, node.value);
			List<DataRow> left = (List<DataRow>) splitted.get(0);
			List<DataRow> right = (List<DataRow>) splitted.get(1);
			
			if (!isLeaf(node.left))
				node.left = prune_pep(node.left, left);
			
			if (!isLeaf(node.right))
				node.right = prune_pep(node.right, right);
			
			return node;
		}
	}
	
	public void train(List<double[]> data, List<String> y) {
		Set<String> targetSet = new HashSet<String>(y);
		targetList = new ArrayList<String>(targetSet);
		Collections.sort(targetList);
		
		List<DataRow> dataSet = transform(data, y);
		
		long size = dataSet.size();
		long cutPoint = size * 9 / 10;
		List<DataRow> train = new ArrayList<DataRow>();
		List<DataRow> test = new ArrayList<DataRow>();
		
		
		
		for (int i = 0; i < size; i ++) {
			if (i < cutPoint) {
				train.add(dataSet.get(i));
			}
			
			else {
				test.add(dataSet.get(i));
			}
		}
		
		if (merge == "None") {
			TreeNode root = createTree(dataSet, 0);
			tree = new BTree(root);
		}
		
		else if (merge == "pep" ) {
			TreeNode root = createTree(dataSet, 0);
			tree = new BTree(root);
			root = prune_pep(root, dataSet);
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
	
	public double[] probaArray(List<DataRow> dataSet) {
		DecimalFormat df = new DecimalFormat( "0.00000 ");
		double[] ret = new double[targetList.size()];
		int size = dataSet.size();
		
		for (int i = 0; i < ret.length; i ++) 
			ret[i] = (double) 0.0;
		
		for (DataRow item : dataSet) {
			String y = item.y;
			int index = targetList.indexOf(y);
			ret[index] += 1;
		}
		
		for (int i = 0; i < ret.length; i ++) {
			ret[i] /= size;
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
		
		List<DataRow> dataSet = new ArrayList<DataRow>();
		dataSet = transform(train, ytrain);
		System.out.println(dataSet.get(0).x.length);
	}
}
