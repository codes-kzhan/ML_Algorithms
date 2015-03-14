import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.lang.Math;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class Kmeans {
   public static String CFILE = "c1.txt";
   public static void main(String[] args) throws Exception {
	   Configuration conf = new Configuration();
	   FileSystem fs = FileSystem.get(conf);
	   String inPath = args[0];
	   String outBase = args[1];
	   
	   for (int i = 0; i < 20; i++) {
		   System.out.println("Iteration : " + i + "================================================");
		   String outDir = outBase + "/" + i;
		   String cenPath = "c1.txt";
		   
		   if (i > 0) {
			   String cenDir = outBase + "/" + (i - 1);
			   Path cenD = new Path(cenDir);
			   cenPath = outBase + "/c/" + i + "-centroid.txt";
			   FileUtil.copyMerge(fs, cenD, fs, new Path(cenPath), false, conf, "");
		   }
		   
		   conf.set(CFILE, cenPath);
		   Job job = new Job(conf, "Kmeans");
		   job.setJarByClass(Kmeans.class);
		   job.setOutputKeyClass(Text.class);
		   job.setOutputValueClass(Text.class);
		   
		   job.setMapOutputKeyClass(IntWritable.class);
		   job.setMapOutputValueClass(Text.class);
	
		   job.setMapperClass(Map.class);
		   job.setReducerClass(Reduce.class);
	
		   job.setInputFormatClass(TextInputFormat.class);
		   job.setOutputFormatClass(TextOutputFormat.class);
	
		   FileInputFormat.addInputPath(job, new Path(inPath));
		   FileOutputFormat.setOutputPath(job, new Path(outDir));
	
		   job.waitForCompletion(true);
	   }
   }
   
   public static double[] parsePoint(String input) {
	   if (input.startsWith("K"))
		   input = input.split("\t")[1];
	   String[] tokens = input.split(" ");
	   double[] point = new double[tokens.length];
	   for (int i = 0; i < point.length; i++) {
		   point[i] = Double.parseDouble(tokens[i]);
	   }
	   return point;
   }
   
   public static double distance(double[] a, double[] b) {
	   double d = 0.0;
	   for (int i = 0; i < a.length; i++) {
		   d += (a[i] - b[i]) * (a[i] - b[i]);
	   }
	   return Math.sqrt(d);
   }
   
   public static String ArrayToString(double[] a) {
	   String result = new String();
	   for (double c : a) {
		   result = result + c + " ";
	   }
	   return result;
   }
   
   public static class Map extends Mapper<LongWritable, Text, IntWritable, Text> {
	   
	  public ArrayList<double[]> centroids = new ArrayList<double[]>();
	  
	  protected void setup(Context context) throws IOException, InterruptedException {
		  FileSystem fs = FileSystem.get(context.getConfiguration());
		  Path cFile = new Path(context.getConfiguration().get(CFILE));
		  DataInputStream fsd = new DataInputStream(fs.open(cFile));
		  BufferedReader reader = new BufferedReader(new InputStreamReader(fsd));
		  String line;
		  while ((line = reader.readLine()) != null) {
			  if (!line.startsWith("C"))
				  centroids.add(parsePoint(line));
		  }
		  reader.close();
	  }
	  
      public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
         double[] point = parsePoint(value.toString());
         double minimum = Long.MAX_VALUE;
         int bestMatch = 0;
         for (double[] c : centroids) {
        	 double tmp = distance(point, c);
        	 if (tmp < minimum) {
        		 minimum = tmp;
        		 bestMatch = centroids.indexOf(c);
        	 }
         }
         String dist = minimum * minimum + "";
         context.write(new IntWritable(bestMatch), new Text(ArrayToString(point)));
         context.write(new IntWritable(-1), new Text(dist));
      }
   }


   public static class Reduce extends Reducer<IntWritable, Text, Text, Text> {
	   
	  public ArrayList<double[]> centroids = new ArrayList<double[]>();
		  
	  protected void setup(Context context) throws IOException, InterruptedException {
		  FileSystem fs = FileSystem.get(context.getConfiguration());
		  Path cFile = new Path(context.getConfiguration().get(CFILE));
		  DataInputStream fsd = new DataInputStream(fs.open(cFile));
		  BufferedReader reader = new BufferedReader(new InputStreamReader(fsd));
		  String line;
		  while ((line = reader.readLine()) != null) {
			  if (!line.startsWith("C"))
				  centroids.add(parsePoint(line));
		  }
		  reader.close();
	  }
	  
      public void reduce(IntWritable key, Iterable<Text> values, Context context)
              throws IOException, InterruptedException {
    	 double cost = 0.0;
    	 double[] new_centroid = new double[centroids.get(0).length];
         if (key.get() == -1) {
        	 for (Text str : values) {
        		 cost += Double.parseDouble(str.toString());
        	 }
        	 System.out.print("Cost : " + cost);
             String c = cost + "";
             context.write(new Text("C" + "-" + key.toString()), new Text(c));
         }
         
         else {
        	 for (int i = 0; i < centroids.get(0).length; i++) {
        		 new_centroid[i] = 0;
        	 }
        	 int nn = 0;
        	 for (Text str : values) {
        		 String[] tokens = str.toString().split(" ");
        		 for (int i = 0; i < tokens.length; i++) {
        			 new_centroid[i] += Double.parseDouble(tokens[i]);
        		 }
        		 nn++;
        	 }
        	 for (int i = 0; i < centroids.get(0).length; i++) {
        		 new_centroid[i] /= nn;
        	 }
        	 context.write(new Text("K" + "-" + key.toString()), new Text(ArrayToString(new_centroid)));
         }
      }
   }
}
