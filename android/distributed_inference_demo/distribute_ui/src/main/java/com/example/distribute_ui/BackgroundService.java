package com.example.distribute_ui;

import static com.example.SecureConnection.Dataset.readCSV;

import android.Manifest;
import android.app.Notification;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Environment;
import android.os.IBinder;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import com.example.SecureConnection.Communication;
import com.example.SecureConnection.Config;
import com.example.SecureConnection.Dataset;
import com.example.SecureConnection.LoadBalance;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.Properties;

public class BackgroundService extends Service {

    private static final int NOTIFICATION_ID = 123;
    private static final String CHANNEL_ID = "MyChannelID";
    public static double[] results;
    public static final String TAG = "Lingual_backend";
    public static final String ACTION_MODEL_DOWNLOADED = "com.example.distributed_ui.DOWNLOADED";
    private String role = "worker";
    private boolean model_exist = false; // depreciated. Don't need.
    private boolean need_monitor = false;

    private boolean running_classification = false;
    private boolean running_generation = true;

    private String getServerIPAddress() {
        String serverIP = "";
        Properties properties = new Properties();
        try {
            InputStream inputStream = getAssets().open("config.properties");
            properties.load(inputStream);
            serverIP = properties.getProperty("server_ip");
            inputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
            // Handle the exception
        }
        return serverIP;
    }

    @Override
    public void onCreate(){
        super.onCreate();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d(TAG, "background service started");
        int id = 0;
        if (intent != null && intent.hasExtra("role")) {
            id = intent.getIntExtra("role", 0);
        }
        if (id == 1) {
            role = "header";
        }
        Log.d(TAG, "role is " + role);

        String modelName = "";
        if (intent != null && intent.hasExtra("model")) {
            modelName = intent.getStringExtra("model");
            System.out.println("model name is: "+ modelName);
        }

        ExecutorService executor = Executors.newSingleThreadExecutor();
        String finalModelName = modelName;
        executor.submit(() -> {
            String server_ip = getServerIPAddress();
            Config cfg = new Config(server_ip, 23456); // set server ip and port
            Communication com = new Communication(cfg);
            Communication.loadBalance = new LoadBalance(com, cfg);
            com.param.modelPath = getFilesDir() + "";
            Log.d(TAG, "Model PATH: " +  com.param.modelPath);

            // 1. send IP to server to request model
            if (role.equals("header")) {
                Log.d(TAG, "test Model");
                need_monitor = com.sendIPToServer(role, finalModelName);
            } else {
                need_monitor = com.sendIPToServer(role, "");
            }

            // 2. Initiate device monitor for server-side optimization
            if (need_monitor) {
                Intent broadcastIntent = new Intent();
                broadcastIntent.setAction("START_MONITOR");
                LocalBroadcastManager.getInstance(this).sendBroadcast(broadcastIntent);
                sendBroadcast(broadcastIntent);
                Log.d(TAG, "broadcast sent by backgroundService");
            }

            System.out.println("Model PATH: " +  com.param.modelPath);

            // 3.1 check whether modelFile already exists, if not we download model from server
//            File modelFile = new File(com.param.modelPath);
//            if (modelFile.exists() && modelFile.isFile()){
//                model_exist = true;
//            }

            // 3.2 start downloading required model and tokenizer files from server
            com.runPrepareThread(model_exist);

            // 4. initiate inference in background service
            com.param.classes = new String[]{"Negative", "Positive"};
            Dataset dataset = null;

            while (com.param.numSample <= 0)
                Thread.sleep(1000);


            String [] test_input = new String[com.param.numSample];
            if (cfg.isHeader()) {
                System.out.println("This is header!");
                System.out.println(com.param.numSample);
                int j = 0;
                while (j < com.param.numSample) {
                    test_input[j++] = "I hate machine learning, what is";
                    test_input[j++] = "I don't know machine learning, what is";
                    test_input[j++] = "I fancy machine learning, what is";
                    test_input[j++] = "I love java and python, which is";
                    test_input[j++] = "University of California Irvine is a public university located in";
                }

                    // 5. option for running inference with dataset. (Experiment use only)
    //                dataset = new Dataset(getFilesDir() + "/wikitext-2-100.csv", 1, com.param.numSample);   // generation dataset
    //                if (dataset.texts.size() == 0){
    //                    System.out.println("No dataset exists, load dataset fail");
    //                }else {
    //                    System.out.println("Load dataset successfully, with size " + dataset.texts.size());
    //                }
    //                test_input = dataset.texts.subList(0, com.param.numSample).toArray(new String[0]);
            }

            System.out.println(test_input);
            System.out.println(com.param.numSample);

            int corePoolSize = 2;
            int maximumPoolSize = 2;
            int keepAliveTime = 500;

            try {
                Log.d(TAG, "communication starts to running");
                com.running(corePoolSize, maximumPoolSize, keepAliveTime, test_input);
            } catch (IOException | InterruptedException e) {
                throw new RuntimeException(e);
            }
            double startTime = System.nanoTime();
            results = com.timeUsage;

            if (running_classification) {
                if (cfg.isHeader()) {
                    double accuracy = 0.0;
                    for (int i = 0; i < com.logits.size(); i++) {
                        int pred = binaryClassify(com.logits.get(i));
                        int truth = dataset.labels.get(i).equals("positive") ? 1 : 0;
                        if (pred == truth) {
                            accuracy += 1;
                        }
                    }
                    Log.d(TAG, "Task Accuracy: " + (accuracy / com.logits.size()));
                }
            }
            Log.d(TAG, "Results Computation Time: " + (System.nanoTime() - startTime)/1000000000.0);
            return null;
        });

        return START_STICKY; // This tells the system to restart the service if it gets killed due to resource constraints.
    }

    private void loadZipFile() {
        File sourceFile = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "device.zip");
        File destinationFile = new File(getFilesDir() + "/device.zip");

        if (!destinationFile.getParentFile().exists()) {
            destinationFile.getParentFile().mkdirs(); // Create the parent path if it doesn't exist
        }

        Log.d(TAG, "SourceFile: " + sourceFile.getAbsolutePath());
        Log.d(TAG, "DestFile: " + destinationFile.getAbsolutePath());

        try (InputStream in = new FileInputStream(sourceFile)) {
            try (OutputStream out = new FileOutputStream(destinationFile)) {
                // Transfer bytes from in to out
                byte[] buf = new byte[1024];
                int len;
                while ((len = in.read(buf)) > 0) {
                    out.write(buf, 0, len);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            // Handle the exception
        }

    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
    @Override
    public void onDestroy(){
        super.onDestroy();
    }

    public native int binaryClassify(byte[] data);

}
