import java.io.IOException;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.conf.Configuration;

public class MaxTemperaturePerYear {

    // Mapper class
  public static class TemperatureMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
    private static final int MISSING = 9999;

    @Override
    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {

        String line = value.toString().trim();
        if (line.isEmpty()) return;

        try {
            String[] fields = line.split("\\s+");
            if (fields.length < 6) return; // skip malformed line

            String dateStr = fields[1];  // e.g., "20200101"
            String year = dateStr.substring(0, 4);

            double tempDouble = Double.parseDouble(fields[5]);

            if (tempDouble != MISSING) {
                int temp = (int) Math.round(tempDouble * 10); // convert to int tenths of degrees
                context.write(new Text(year), new IntWritable(temp));
            }
        } catch (Exception e) {
            // skip malformed line or parse errors
        }
    }
}


    // Reducer class
    public static class MaxTemperatureReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            
            int maxTemp = Integer.MIN_VALUE;

            for (IntWritable val : values) {
                maxTemp = Math.max(maxTemp, val.get());
            }

            context.write(key, new IntWritable(maxTemp));
        }
    }

    // Driver class
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: MaxTemperaturePerYear <input path> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Max Temperature Per Year");

        job.setJarByClass(MaxTemperaturePerYear.class);
        job.setMapperClass(TemperatureMapper.class);
        job.setReducerClass(MaxTemperatureReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
