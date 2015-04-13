package RandomForest;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class util {
	
	public static List loadData(String filename) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(filename));
		in.readLine();
		String line;
		StringBuilder sb = new StringBuilder();
		List<String> target = new ArrayList<String>();
		List<double[]> features = new ArrayList<double[]>();
		
		while ((line = in.readLine()) != null) {
			String item[] = line.split(",");
			int length = item.length - 1;
			String last = item[item.length-1];
			target.add(last);
			double feat[] = new double[length];
			
			for (int i = 0; i < length; i++) {
				feat[i] = Double.parseDouble(item[i]);
			}
			
			features.add(feat);
		}
		
		List ret = new ArrayList();
		ret.add(features);
		ret.add(target);
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
	
	public static double gini(List<String> y) {
		double ret = 0;
		Map<String, Double> yDict = new HashMap<String, Double>();
		
		for (String i : y) {
			if (!yDict.containsKey(i))
				yDict.put(i,  (double) 0);
			double curr = yDict.get(i);
			yDict.put(i, curr + 1);
		}
		
		for (String i : yDict.keySet()) {
			double curr = yDict.get(i);
			ret += (curr / y.size()) * (curr / y.size());
		}
		
		return 1 - ret;
	}
	
	public static String classify(List<String> y) {
		Map<String, Double> yDict = new HashMap<String, Double>();
		
		for (String i : y) {
			if (!yDict.containsKey(i))
				yDict.put(i,  (double) 0);
			double curr = yDict.get(i);
			yDict.put(i, curr + 1);
		}
		
		double Max = (double) 0;
		String maxId = null;
		
		for (String i : yDict.keySet()) {
			double curr = yDict.get(i);
			if (curr > Max) {
				Max = curr;
				maxId = i;
			}
		}
		
		return maxId;
	}
	
	public static List bootstrap(List<double[]> data, List<String> y) {
		int length = y.size();
		List<double[]> dataTrain =  new ArrayList<double[]>();
		List<String> yTrain = new ArrayList<String>();
		Random r = new Random();
		
		for (int i = 0; i < length; i++) {
			int index = r.nextInt(length);
			dataTrain.add(data.get(index));
			yTrain.add(y.get(index));
		}
		
		List ret = new ArrayList();
		ret.add(dataTrain);
		ret.add(yTrain);
		return ret;
	}
	
	public static int argmax(double[] row) {
		int maxId = 0;
		double max = row[0];
		
		for (int i = 1; i < row.length; i++) {
			double tmp = row[i];
			if (tmp > max) {
				max = tmp;
				maxId = i;
			}
		}
		
		return maxId;
	}
	
	public static double multiLogLoss(List<double[]> yPred, List<String> yTrue, double eps) {
		List<String> targetList = new ArrayList<String>();
		Set<String> targetSet = new HashSet<String>(yTrue);
		targetList = new ArrayList<String>(targetSet);
		Collections.sort(targetList);
		
		for (int i = 0; i < yPred.size(); i++) {
			double sum = 0.0;
			for (int j = 0; j < yPred.get(i).length; j++) {
				if (yPred.get(i)[j] < eps)
					yPred.get(i)[j] = eps;
				else if (yPred.get(i)[j] > 1 - eps)
					yPred.get(i)[j] = 1 - eps;
				sum += yPred.get(i)[j];
			}
			
			for (int j = 0; j < yPred.get(i).length; j++) {
				yPred.get(i)[j] /= sum;
			}
		}
		
		int nFactors = yPred.get(0).length;
		int nSamples = yPred.size();
		double vectSum = 0.0;
		
		for (int i = 0; i < nSamples; i++) {
			double[] current = new double[nFactors];
			int index = targetList.indexOf(yTrue.get(i));
			current[index] = 1.0;
			for (int j = 0; j < nFactors; j++) {
				vectSum += current[j] * Math.log(yPred.get(i)[j]);
			}
		}
		
		return -1.0 / nSamples * vectSum;
	}
}
