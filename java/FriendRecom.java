import java.io.IOException;
import java.util.*;
import java.lang.Math;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class FriendRecom {
   public static void main(String[] args) throws Exception {
	   Configuration conf = new Configuration();

	   Job job = new Job(conf, "FriendRecom");
	   job.setJarByClass(FriendRecom.class);
	   job.setOutputKeyClass(Text.class);
	   job.setOutputValueClass(Text.class);
	   
	   job.setMapOutputKeyClass(Text.class);
	   job.setMapOutputValueClass(Text.class);

	   job.setMapperClass(Map.class);
	   job.setReducerClass(Reduce.class);

	   job.setInputFormatClass(TextInputFormat.class);
	   job.setOutputFormatClass(TextOutputFormat.class);

	   FileInputFormat.addInputPath(job, new Path(args[0]));
	   FileOutputFormat.setOutputPath(job, new Path(args[1]));

	   job.waitForCompletion(true);
   }
   
   public static class Map extends Mapper<LongWritable, Text, Text, Text> {
      public void map(LongWritable key, Text value, Context context) 
    		  throws IOException, InterruptedException {
         String line = value.toString();
         int index1 = line.indexOf(',');
         int index2 = line.indexOf('\t');
         if (index1 == -1)
        	 return;
         if (index2 == -1)
        	 return;
         String user_id = line.substring(0, index1);
         String friend_id = line.substring(index1 + 1, index2);
         String number = line.substring(index2 + 1);
         context.write(new Text(user_id), new Text(friend_id + "," + number));
      }
   }


   public static class Reduce extends Reducer<Text, Text, Text, Text> {
      public void reduce(Text key, Iterable<Text> values, Context context)
              throws IOException, InterruptedException {
         Vector<int[]> numbers = new Vector<int[]>();
         for (Text val : values) {
            String val_str = val.toString();
            int index = val_str.indexOf(",");
            if (index == -1)
            	return;
            String friend_id = val_str.substring(0, index);
            String number = val_str.substring(index + 1);
            int[] pair = new int[2];
            pair[0] = Integer.parseInt(friend_id);
            pair[1] = Integer.parseInt(number);
            numbers.add(pair);
         }
         for (int i=0; i<numbers.size(); i++) {
        	 int max = numbers.get(i)[1], maxId = i;
        	 for (int j=i; j<numbers.size(); j++) {
        		 if (numbers.get(j)[1] > max) {
        			 max = numbers.get(j)[1];
        			 maxId = j;
        		 }
        	 }
        	 Collections.swap(numbers, maxId, i);
         }
         if (numbers.size() > 10) {
        	 for (int i=0; i<10; i++) {
        		 String ret = numbers.get(i)[0] + ":" + numbers.get(i)[1];
        		 context.write(key, new Text(ret));
        	 }
         }
         if (numbers.size() <= 10) {
        	 for (int i=0; i<numbers.size(); i++) {
        		 String ret = numbers.get(i)[0] + ":" + numbers.get(i)[1];
        		 context.write(key, new Text(ret));
        	 }
         }
      }
   }
}
