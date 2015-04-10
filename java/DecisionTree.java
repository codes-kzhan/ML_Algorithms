package RandomForest;

import java.io.*;
import java.util.*;


class TreeNode {
	int feature;
	float value;
	String maxClass;
/*	Map<String, Float> probas = new HashMap<String, Float>();*/
	TreeNode left = null;
	TreeNode right = null;
	
	public TreeNode(int feature, float value, String maxClass) {
		this.feature = feature;
		this.value = value;
		this.maxClass = maxClass;
	}
	
	public TreeNode(int feature, float value, String maxClass, TreeNode left, TreeNode right) {
		this.feature = feature;
		this.value = value;
		this.maxClass = maxClass;
/*		this.probas = probas;*/
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
	
	public String getClass(float[] x, TreeNode node) {
		
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
}


public class DecisionTree {
	
	static double tol = 0.0001;
	static int leastSample = 1;
	static int maxDepth = 999;
	static String merge = "pep";
	static int maxFeatures = 10;
	static BTree tree = new BTree();

	public static List loadData(String filename) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(filename));
		in.readLine();
		String line;
		StringBuilder sb = new StringBuilder();
		List<String> target = new ArrayList<String>();
		List<float[]> features = new ArrayList<float[]>();
		
		while ((line = in.readLine()) != null) {
			String item[] = line.split(",");
			int length = item.length - 1;
			String last = item[item.length-1];
			target.add(last);
			float feat[] = new float[length];
			
			for (int i = 0; i < length; i++) {
				feat[i] = Float.parseFloat(item[i]);
			}
			
			features.add(feat);
		}
		
		List ret = new ArrayList();
		ret.add(features);
		ret.add(target);
		return ret;
	}
	
	public DecisionTree(float tol, int leastSample, int maxDepth, String merge, int maxFeatures) {
		this.tol = tol;
		this.leastSample = leastSample;
		this.maxDepth = maxDepth;
		this.merge = merge;
		this.maxFeatures = maxFeatures;
	}
	
	public DecisionTree() {}
	
	public static float gini(List<String> y) {
		float ret = 0;
		Map<String, Float> yDict = new HashMap<String, Float>();
		
		for (String i : y) {
			if (!yDict.containsKey(i))
				yDict.put(i,  (float) 0);
			float curr = yDict.get(i);
			yDict.put(i, curr + 1);
		}
		
		for (String i : yDict.keySet()) {
			float curr = yDict.get(i);
			ret += (curr / y.size()) * (curr / y.size());
		}
		
		return 1 - ret;
	}
	
	public static String classify(List<String> y) {
		Map<String, Float> yDict = new HashMap<String, Float>();
		
		for (String i : y) {
			if (!yDict.containsKey(i))
				yDict.put(i,  (float) 0);
			float curr = yDict.get(i);
			yDict.put(i, curr + 1);
		}
		
		float Max = (float) 0;
		String maxId = null;
		
		for (String i : yDict.keySet()) {
			float curr = yDict.get(i);
			if (curr > Max) {
				Max = curr;
				maxId = i;
			}
		}
		
		return maxId;
	}
	
	public static List binSplitDataSet(List<float[]> data, List<String> y, int feature, float value) {
		List ret = new ArrayList();
		List<float[]> mat0 = new ArrayList<float[]>();
		List<float[]> mat1 = new ArrayList<float[]>();
		List<String> y0 = new ArrayList<String>();
		List<String> y1 = new ArrayList<String>();
		
		for (int i = 0; i < data.size(); i++) {
			float[] row = data.get(i);
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
	
	public static List chooseBestSplit(List<float[]> data, List<String> y) {
		List ret = new ArrayList();
		Set<String> ySet = new HashSet<String>(y);
		
		if (ySet.size() == 1) {
			ret.add(null); ret.add(null); ret.add(y.get(0));
			return ret;
		}
		
		int n = data.get(0).length;
		float S = gini(y);
		float bestS = (float) 1.0;
		int bestIndex = 0;
		float bestValue = (float) 0.0;
		List<Integer> featureList = new ArrayList<Integer>();
		int visitedFeatures = 0;
		
		for (int i = 0; i < n; i++)
			featureList.add(i);
		
		while (visitedFeatures < maxFeatures && featureList.size() > 0) {
			Random r = new Random();
			int randNum = r.nextInt(featureList.size());
			int featIndex = featureList.get(randNum);
			featureList.remove(randNum);
			List<Float> column = new ArrayList<Float>();
			
			for (float[] i : data) 
				column.add(i[featIndex]);
			
			Set<Float> featSet = new HashSet<Float>(column);
			if (featSet.size() == 1)
				continue;
			
			visitedFeatures += 1;
			float lastS = (float) 1.0;
			
			for (float splitVal : featSet) {
				List splitted = binSplitDataSet(data, y, featIndex, splitVal);
				List<float[]> mat0 = (ArrayList<float[]>) splitted.get(0);
				List<String> y0 = (ArrayList<String>) splitted.get(1);
				List<float[]> mat1 = (ArrayList<float[]>) splitted.get(2);
				List<String> y1 = (ArrayList<String>) splitted.get(3);
				
				if (mat0.size() < leastSample || mat1.size() < leastSample)
					continue;
				
				float r0 = ((float) y0.size()) / y.size();
				float r1 = ((float) y1.size()) / y.size();
				float newS = r0 * gini(y0) + r1 * gini(y1);
				
				if (newS > lastS)
					break;
				
				if (newS < bestS) {
					bestIndex = featIndex;
					bestValue = splitVal;
					bestS = newS;
				}
				
				lastS = newS;
			}
		}
		
		if (S - bestS < tol) {
			ret.add(null); ret.add(null); ret.add(classify(y));
			return ret;
		}
		
		List splitted = binSplitDataSet(data, y, bestIndex, bestValue);
		List<float[]> mat0 = (ArrayList<float[]>) splitted.get(0);
		List<String> y0 = (ArrayList<String>) splitted.get(1);
		List<float[]> mat1 = (ArrayList<float[]>) splitted.get(2);
		List<String> y1 = (ArrayList<String>) splitted.get(3);
		
		if (mat0.size() < leastSample || mat1.size() < leastSample) {
			ret.add(null); ret.add(null); ret.add(classify(y));
			return ret;
		}
		
		ret.add(bestIndex); ret.add(bestValue); ret.add(classify(y));
		return ret;
	}
	
	public static TreeNode createTree(List<float[]> data, List<String> y, int depth) {
		TreeNode retTree = new TreeNode(-1, -1, "Nan");
		
		if (depth > maxDepth) {
			retTree.maxClass = classify(y);
			return retTree;
		}
		
		List bestSplit = chooseBestSplit(data, y);
		Object feat = bestSplit.get(0); Object val = bestSplit.get(1); String mc = (String) bestSplit.get(2);
		if (feat == null) {
			retTree.maxClass = mc;
			return retTree;
		}
		
		retTree.feature = (int) feat;
		retTree.value = (float) val;
		retTree.maxClass = mc;
		List splitted = binSplitDataSet(data, y,  (int) feat, (float) val);
		List<float[]> left = (ArrayList<float[]>) splitted.get(0);
		List<String> yleft = (ArrayList<String>) splitted.get(1);
		List<float[]> right = (ArrayList<float[]>) splitted.get(2);
		List<String> yright = (ArrayList<String>) splitted.get(3);
		retTree.left = createTree(left, yleft, depth + 1);
		retTree.right = createTree(right, yright, depth + 1);
		
		return retTree;
	}
	
	public static boolean isLeaf(TreeNode node) {
		return (node.left == null);
	}
	
	public static TreeNode prune_pep(TreeNode node, List<float[]> data, List<String> y) {
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
			List<float[]> left = (ArrayList<float[]>) splitted.get(0);
			List<String> yleft = (ArrayList<String>) splitted.get(1);
			List<float[]> right = (ArrayList<float[]>) splitted.get(2);
			List<String> yright = (ArrayList<String>) splitted.get(3);
			
			if (!isLeaf(node.left))
				node.left = prune_pep(node.left, left, yleft);
			
			if (!isLeaf(node.right))
				node.right = prune_pep(node.right, right, yright);
			
			return node;
		}
	}
	
	public static void train(List<float[]> data, List<String> y) {
		long length = y.size();
		long cutPoint = length * 9 / 10;
		List<String> ytrain = new ArrayList<String>();
		List<float[]> train = new ArrayList<float[]>();
		List<String> ytest = new ArrayList<String>();
		List<float[]> test = new ArrayList<float[]>();
		
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
	
	public static List predict(List<float[]> data) {
		List<String> ret = new ArrayList<String>();
		
		for (float[] x : data) {
			String pred = tree.getClass(x, tree.root);
			ret.add(pred);
		}
		
		return ret;
	}
	
	public static long correctNum(List<String> y, List<String> yPred) {
		long ret = 0;
		
		for (int i = 0; i < y.size(); i++) {
			if (y.get(i).equals(yPred.get(i)))
				ret += 1;
		}
		
		return ret;
	}
	
	public static double score(List<float[]> data, List<String> y) {
		List<String> yPred = new ArrayList<String>();
		yPred = predict(data);
		return correctNum(y, yPred) / ( (double) y.size() );
	}
	
	public static void main(String[] args) throws IOException  {
		String filename = "/home/yejiming/desktop/python/ML_Algorithms/dataset.csv";
		List ret = loadData(filename);
		List<float[]> data = (List<float[]>) ret.get(0);
		List<String> y = (List<String>) ret.get(1);
		List<String> ytrain = new ArrayList<String>();
		List<float[]> train = new ArrayList<float[]>();
		List<String> ytest = new ArrayList<String>();
		List<float[]> test = new ArrayList<float[]>();
		
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
		
		cart learner = new DecisionTree((float) (tol=0.0001), leastSample=1, maxDepth=999, merge="pep", maxFeatures=93);
		learner.train(train, ytrain);
		double s = learner.score(test, ytest);
/*		List<String> yPred = new ArrayList<String>();
		yPred = learner.predict(test);
		System.out.println(yPred.get(0).equals(ytest.get(0)));*/
		System.out.println(s);
	}
}
