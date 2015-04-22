package RandomForest2;

import java.io.*;
import java.util.*;

class DataRow {
	
	double[] x;
	String y;
	
	public DataRow(double[] x, String y) {
		this.x = x;
		this.y = y;
	}
}

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
	
	public static List<DataRow> transform (List<double[]> data, List<String> y) {
		List<DataRow> ret = new ArrayList<DataRow>();
		int size = y.size();
		for (int i = 0; i < size; i++) {
			DataRow row = new DataRow(data.get(i), y.get(i));
			ret.add(row);
		}
		return ret;
	}
	
	public static List inverse(List<DataRow> dataSet) {
		List ret = new ArrayList();
		List<double[]> data = new ArrayList<double[]>();
		List<String> y = new ArrayList<String>();
		for (DataRow item : dataSet) {
			double[] x = item.x; String yi = item.y;
			data.add(x); y.add(yi);
		}
		ret.add(data); ret.add(y);
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
	
	public static List loadTest(String filename) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(filename));
		in.readLine();
		String line;
		StringBuilder sb = new StringBuilder();
		List<String> ids = new ArrayList<String>();
		List<double[]> features = new ArrayList<double[]>();
		
		while ((line = in.readLine()) != null) {
			String item[] = line.split(",");
			int length = item.length - 1;
			String first = item[0];
			ids.add(first);
			double feat[] = new double[length];
			
			for (int i = 0; i < length; i++) {
				feat[i] = Double.parseDouble(item[i + 1]);
			}
			
			features.add(feat);
		}
		
		List ret = new ArrayList();
		ret.add(features);
		ret.add(ids);
		return ret;
	}
	
	public static double gini(List<DataRow> dataSet) {
		double ret = 0;
		int size = dataSet.size();
		Map<String, Double> yDict = new HashMap<String, Double>();
		
		for (DataRow item: dataSet) {
			String y = item.y;
			if (!yDict.containsKey(y))
				yDict.put(y,  (double) 0);
			double curr = yDict.get(y);
			yDict.put(y, curr + 1);
		}
		
		for (String i : yDict.keySet()) {
			double curr = yDict.get(i);
			ret += (curr / size) * (curr / size);
		}
		
		return 1 - ret;
	}
	
	public static String classify(List<DataRow> dataSet) {
		Map<String, Double> yDict = new HashMap<String, Double>();
		
		for (DataRow item : dataSet) {
			String y = item.y;
			if (!yDict.containsKey(y))
				yDict.put(y,  (double) 0);
			double curr = yDict.get(y);
			yDict.put(y, curr + 1);
		}
		
		double Max = (double) 0;
		String maxId = null;
		
		for (String y : yDict.keySet()) {
			double curr = yDict.get(y);
			if (curr > Max) {
				Max = curr;
				maxId = y;
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
	
	public static void submit(List<double[]> yPred, List<String> ids, String filename) throws IOException {
		File file = new File(filename);
		FileOutputStream fs = new FileOutputStream(file);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fs));
		bw.write("id,class_1,class_2,class_3,class_4,class_5,class_6,class_7,class_8,class_9\n");
		
		for (int i = 0; i < yPred.size(); i++) {
			String prediction = "";
			double[] row = yPred.get(i);
			for (int j = 0; j < row.length; j++) {
				prediction += row[j];
				if (j == row.length - 1)
					prediction += "\n";
				else
					prediction += ",";
			}
			bw.write(ids.get(i) + "," + prediction);
		}
		bw.flush();
	}
}
