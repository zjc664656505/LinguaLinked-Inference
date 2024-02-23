package com.example.SecureConnection;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Dataset {
    private int BATCH_SIZE = 4;
    private String CSV_PATH;
    public List<String> texts;
    public List<String> labels;

    public int size;

    public Dataset(String path, int batch_size, int number) {
        this.texts = new ArrayList<>();
        this.labels = new ArrayList<>();
        this.CSV_PATH = path;
        this.BATCH_SIZE = batch_size;
        this.size = number;

        try (CSVReader reader = new CSVReader(new FileReader(CSV_PATH))) {
            String[] line;
            int num = 0;
            while ((line = reader.readNext()) != null) {
                texts.add(line[0]);
                labels.add(line[1]);
                num++;
                if (num > size)
                    break;
            }
            // csv column header
            texts.remove(0);
            labels.remove(0);
        } catch (CsvValidationException | IOException e) {
            e.printStackTrace();
        }

    }

//    private static List<Integer>  batchEncode(){
//        List<DataSample> dataPoints = readCSV(CSV_PATH);
//        for (int i = 0; i < dataPoints.size(); i += BATCH_SIZE) {
//            List<DataSample> batch = new ArrayList<>();
//            for (int j = 0; j < BATCH_SIZE && i + j < dataPoints.size(); j++) {
//                batch.add(dataPoints.get(i + j));
//            }
//            // Here, you can pass the batch to your model for inference.
//            // For demonstration purposes, we'll just print the batch.
//            System.out.println("Batch:");
//            for (DataSample dp : batch) {
//                System.out.println(dp);
//            }
//        }
//    }

//    public static List<String[]> readCSVToBatch(String path, int count) {
//        List<String[]>  data = new ArrayList<>();
//        String[] text = new String[count];
//        String[] label = new String[count];
//
//        int i = 0;
//        try (CSVReader reader = new CSVReader(new FileReader(path))) {
//            String[] line;
//            while (((line = reader.readNext()) != null) && (i <= count)) {
//                texts.add(line[0]);
//                labels.add(line[1]);
//                i += 1;
//            }
//        } catch (CsvValidationException | IOException e) {
//            e.printStackTrace();
//        }
//        return texts, labels;
//    }


    public static List<DataSample> readCSV(String path, int count) {
        List<DataSample> dataPoints = new ArrayList<>();
        int i = 0;

        try (CSVReader reader = new CSVReader(new FileReader(path))) {
            String[] line;
            while (((line = reader.readNext()) != null) && (i <= count)) {
                dataPoints.add(new DataSample(line[0], line[1]));
                i += 1;
            }
        } catch (CsvValidationException | IOException e) {
            e.printStackTrace();
        }
        return dataPoints;
    }

    public static class DataSample {
        String input;
        String label;
        DataSample(String input, String label) {
            this.input = input;
            this.label = label;
        }
        @Override
        public String toString() {
            return "Input: " + input + ", Label: " + label;
        }
    }

    public static class DatchSampleID {
        int[][] input_id;
        int[][] label;

        DatchSampleID (int[][] input_id, int[][] label) {
            this.input_id = input_id;
            this.label = label;
        }
        @Override
        public String toString() {
            return "Input id: " + Arrays.toString(input_id) + ", Label: " + Arrays.toString(label);
        }
    }

}