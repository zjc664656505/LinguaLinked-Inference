package com.example.SecureConnection;

import static com.example.distribute_ui.BackgroundService.TAG;

import android.os.Build;
import android.util.Log;

import com.example.distribute_ui.BackgroundService;
import com.opencsv.CSVReader;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.Semaphore;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.json.JSONObject;
import org.json.JSONArray;
import org.json.JSONTokener;

public class Utils {

    public static void unzipFile(String filePath) {
        File file = new File(filePath);
        if (!file.exists()) {
            Log.d(TAG, "The zip file '" + filePath + "' does not exist.");
            return;
        }

        try {
            ZipFile zipFile = new ZipFile(file);
            zipFile.close();
        } catch (IOException e) {
            Log.d(TAG, "'" + filePath + "' is not a zip file. (Packet loss)");
            e.printStackTrace();
            Log.d(TAG, "zip error: " + e.toString());
        }

        String targetDirectory = file.getParent() + "/device";
        unzip(filePath, targetDirectory);
        Log.d(TAG, "Zip file extracted to "+ targetDirectory +" successfully.");
    }

    private static void unzip(String zipFilePath, String destDir) {
        File dir = new File(destDir);
        // create output directory if it doesn't exist
        if(!dir.exists()) dir.mkdirs();

        FileInputStream fis;
        //buffer for read and write data to file
        byte[] buffer = new byte[1024];
        try {
            fis = new FileInputStream(zipFilePath);
            ZipInputStream zis = new ZipInputStream(fis);
            ZipEntry ze = zis.getNextEntry();
            while(ze != null){
                String fileName = ze.getName();
//                if (fileName.endsWith(".json") || fileName.endsWith(".onnx")) {
                if (!ze.isDirectory()){
                    File newFile = new File(destDir + File.separator + fileName);
                    System.out.println("Unzipping to "+ newFile.getAbsolutePath());
                    new File(newFile.getParent()).mkdirs();
                    FileOutputStream fos = new FileOutputStream(newFile);
                    int len;
                    while ((len = zis.read(buffer)) > 0) {
                        fos.write(buffer, 0, len);
                    }
                    fos.close();
                }
                //close this ZipEntry
                zis.closeEntry();
                ze = zis.getNextEntry();
            }
            //close last ZipEntry
            zis.closeEntry();
            zis.close();
            fis.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static byte[] convertIntToByteArray(int value) {
        byte[] bytes = new byte[Integer.BYTES];
        int length = bytes.length;
        for (int i = 0; i < length; i++) {
            bytes[length - i - 1] = (byte) (value & 0xFF);
            value >>= 8;
        }
        ;
        return bytes;
    }

    public static int convertByteArrayToInt(byte[] bytes) {
        int value = 0;
        for(byte b :bytes)
        {
            value = (value << 8) + (b & 0xFF);
        }
        return value;
    }

    public static byte[] convertIntArrayToByteArray(int[] value) {
        ByteBuffer byteBuffer = ByteBuffer.allocate(value.length * 4);
        for (int i : value) {
            byteBuffer.putInt(i);
        }
        return byteBuffer.array();
    }

    public static int[] convertByteArrayToIntArray(byte[] bytes) {
        if (bytes.length % 4 != 0) {
            throw new IllegalArgumentException("byte array length should be a multiple of 4");
        }

        int[] ints = new int[bytes.length / 4];
        ByteBuffer byteBuffer = ByteBuffer.wrap(bytes);
        for (int i = 0; i < ints.length; i++) {
            ints[i] = byteBuffer.getInt();
        }
        return ints;
    }


    public static ArrayList<Integer> convertIntegerArrayToArrayList(int[] data){
        ArrayList<Integer> res = new ArrayList<>();
        for (int i: data){
            res.add(i);
        }
        return res;
    }

    public static int[] convertArrayListToIntArray(ArrayList<Integer> data){
//        return data.toArray(new Integer[data.size()]);
        return data.stream().mapToInt(Integer::intValue).toArray();
    }

    public static void await(Semaphore latch,  int numOfPerimits) throws InterruptedException {
        // Try to mimic latch await. See if all Semaphore permits can be acquired.
        System.out.println("Available latch: " + (numOfPerimits - latch.availablePermits()));

        for (int i = 0; i < numOfPerimits; i++) {
            latch.acquire();  // Try to mimic latch await. See if all
            System.out.println("Num of latch: " + i);
        }
//        if (latch.availablePermits() == numOfPerimits)

    }


    public static void acquireLocks(Semaphore latch,  int numOfPerimits) throws InterruptedException {
        // Try to mimic latch await. See if all Semaphore permits can be acquired.
//        for (int i = 0; i < (latch.availablePermits() - numOfPerimits); i++) {
//            latch.acquire();  // Try to mimic latch await. See if all
//            System.out.println("acquire locks: " + i);
//        }
//        if (latch.availablePermits() == numOfPerimits)
        latch.acquire((latch.availablePermits() - numOfPerimits));
    }

    public long calculateFactorial(int n) {
        if (n == 0 || n == 1) {
            return 1;
        } else {
            return n * calculateFactorial(n - 1);
        }
    }


    public void evaluation() {

        System.out.println("Hardware build " + Build.HARDWARE);

        System.out.println("CPU Arch " + System.getProperty("os.arch"));
        System.out.println("CPU ABI " + android.os.Build.CPU_ABI);

        int numCores = Runtime.getRuntime().availableProcessors();

        long startTestTime = System.nanoTime(); // Get the start time

        int numberToCalculate = 20; // You can adjust the number here for more or less intensity
        long result = calculateFactorial(numberToCalculate);
        System.out.println("Factorial of " + numberToCalculate + " is " + result);


        long endTestTime = System.nanoTime(); // Get the end time

        // Calculate the CPU clock speed in GHz
        double elapsedSeconds = (endTestTime - startTestTime) / 1e9; // Convert nanoseconds to seconds
        double cpuClockSpeed = 1.0 / elapsedSeconds; // In GHz

        // Now, cpuClockSpeed contains an estimate of the CPU clock speed.

        System.out.println("Num Cores: " + numCores);
        System.out.println("Clock speed: " + cpuClockSpeed);


        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            String[] supportedAbis = Build.SUPPORTED_ABIS;
            StringBuilder cpuInfo = new StringBuilder();

            for (String abi : supportedAbis) {
                // Use the 'cat' command to read the CPU information from the appropriate file
                String cmd = "/system/bin/cat /sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq";
                try {
                    Process process = Runtime.getRuntime().exec(cmd);
                    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        cpuInfo.append("CPU Max Frequency: ").append(line).append(" KHz\n");
                    }
                    process.waitFor();
                    reader.close();
                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                }
            }
            System.out.println(cpuInfo.toString());


            String cpuSpeed = "Unknown";
            try {
                RandomAccessFile reader = new RandomAccessFile("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r");
                String load = reader.readLine();

                if (load != null) {
                    double speed = Long.parseLong(load) / 1000; // Convert to MHz
                    cpuSpeed = String.valueOf(speed) + " MHz";
                }
                reader.close();

                System.out.println("cpuSpeed " + cpuSpeed);

            } catch (IOException e) {
                e.printStackTrace();
            }


            String maxCpuFreq = "Unknown";
            try {
                RandomAccessFile reader = new RandomAccessFile("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
                String maxFreq = reader.readLine();

                if (maxFreq != null) {
                    double maxSpeed = Long.parseLong(maxFreq) / 1000; // Convert to MHz
                    maxCpuFreq = String.valueOf(maxSpeed) + " MHz";
                }

                reader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println("maxCpuFreq: " + maxCpuFreq);
        } else {
            System.out.println("CPU information not available on this device.");
        }


    }


    public static JSONObject loadJsonFile(String filePath) throws Exception {
        JSONObject json = null;
        try {
            FileReader fileReader = new FileReader(filePath);

            StringBuilder fileContents = new StringBuilder();
            int character;
            while ((character = fileReader.read()) != -1) {
                fileContents.append((char) character);
            }

            JSONTokener jsonTokener = new JSONTokener(fileContents.toString());
            json = new JSONObject(jsonTokener);
            fileReader.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
        return json;
    }
    public static void sortElements(TreeMap<String, ArrayList<JSONObject>> index) throws JSONException {
        for (Map.Entry<String, ArrayList<JSONObject>> entry : index.entrySet()) {
            ArrayList<JSONObject> tmp = new ArrayList<>();
            for (JSONObject obj: entry.getValue()) {
                JSONObject sortedJson = new JSONObject();
                TreeSet<String> keys = new TreeSet<>();
                Iterator<String> keyIter = obj.keys();
                while (keyIter.hasNext())
                    keys.add(keyIter.next());
                for (String key : keys)
                    sortedJson.put(key, obj.getJSONArray(key));
                tmp.add(sortedJson);
            }
            index.put(entry.getKey(), tmp);
        }
    }
    public static int[] JsonArray2IntArray(JSONArray jsonArray){
        int[] intArray = new int[jsonArray.length()];
        for (int i = 0; i < intArray.length; ++i) {
            intArray[i] = jsonArray.optInt(i);
        }
        return intArray;
    }


    public static void getOnnxSession(String modelPath) {

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        System.out.println("Create Environment");

        try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {

            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);

            System.out.println("Loading model from " + modelPath);

            try (OrtSession session = env.createSession(modelPath, opts)) {

                System.out.println("Inputs:");
                for (NodeInfo i : session.getInputInfo().values()) {
                    System.out.println(i.toString());
                }

                System.out.println("Outputs:");
                for (NodeInfo i : session.getOutputInfo().values()) {
                    System.out.println(i.toString());
                }
            }
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }

//        SparseData data = load(args[1]);

//        float[][][][] testData = new float[1][1][28][28];
//        float[][] testDataSKL = new float[1][780];
//
//        String inputName = session.getInputNames().iterator().next();
//
//        for (int i = 0; i < data.labels.length; i++) {
//            if (args.length == 3) {
//                writeDataSKL(testDataSKL, data.indices.get(i), data.values.get(i));
//            } else {
//                writeData(testData, data.indices.get(i), data.values.get(i));
//            }
//
//            try (OnnxTensor test =
//                         OnnxTensor.createTensor(env, args.length == 3 ? testDataSKL : testData);
//                 Result output = session.run(Collections.singletonMap(inputName, test))) {
//
//                int predLabel;
//                if (args.length == 3) {
//                    long[] labels = (long[]) output.get(0).getValue();
//                    predLabel = (int) labels[0];
//                } else {
//                    float[][] outputProbs = (float[][]) output.get(0).getValue();
//                    predLabel = pred(outputProbs[0]);
//                }
//
//            }
//        }
    }

    public static class LBPause {
        public boolean condition = false;

        public synchronized void waitForCondition() throws InterruptedException {
            while (condition) {
                // Wait until the condition becomes true
                wait();
            }
        }
        public synchronized void setConditionTrue() {
            condition = true;
            // Notify any waiting threads that the condition has changed
            notifyAll();
        }

        public synchronized void setConditionFalse() {
            condition = false;
            // Notify any waiting threads that the condition has changed
            notifyAll();
        }
    }
}

