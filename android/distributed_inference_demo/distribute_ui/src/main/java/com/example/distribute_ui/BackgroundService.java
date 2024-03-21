package com.example.distribute_ui;
import android.app.Service;
import android.content.Intent;
import android.os.Environment;
import android.os.IBinder;
import android.util.Log;
import androidx.annotation.Nullable;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;
import com.example.SecureConnection.Communication;
import com.example.SecureConnection.Config;
import com.example.SecureConnection.Dataset;
import com.example.SecureConnection.LoadBalance;
import org.greenrobot.eventbus.EventBus;
import org.greenrobot.eventbus.Subscribe;
import org.greenrobot.eventbus.ThreadMode;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.Properties;

public class BackgroundService extends Service {
    public static double[] results;
    public static final String TAG = "Lingual_backend";
    private String role = "worker";
    private boolean need_monitor = false;
    private final boolean running_classification = false;
    private boolean shouldStartInference = false;
    private boolean runningStatus = false;
    private boolean messageStatus = false;
    public static boolean isServiceRunning = false;

    private String messageContent = "";

    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onRunningStatus(Events.RunningStatusEvent event){
        runningStatus = event.isRunning;
        System.out.println("Running Status is: "+runningStatus);
    }

    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onMessageSentEvent(Events.messageSentEvent event) {
        messageStatus = event.messageSent;
        messageContent = event.messageContent;
    }

    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onEnterChatEvent(Events.enterChatEvent event) {
        shouldStartInference = event.enterChat;
    }

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

    private boolean isModelDirectoryEmpty(String modelPath) {
        File modelDir = new File(modelPath + "/device");
        if (modelDir.isDirectory()) {
            String[] files = modelDir.list();
            return files == null || files.length == 0;
        }
        // Return true if it's not a directory, indicating "empty" in this context.
        return true;
    }

    private void updateIsDirEmpty(boolean isDirEmpty) {
        // Update the repository with the new value
        DataRepository.INSTANCE.setIsDirEmpty(isDirEmpty);
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

            // k is parameter for top-k
            // initial_temp is the parameter for temperature.
            Config cfg = new Config(server_ip, 23456, 7, 0.7f);

            Communication com = new Communication(cfg);
            Communication.loadBalance = new LoadBalance(com, cfg);
            com.param.modelPath = getFilesDir() + "";

            // 1. send IP to server to request model
            if (role.equals("header")) {
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

            // 3.1 start downloading required model and tokenizer files from server
            com.runPrepareThread();

            // 3.2 Check whether the model file exists
            while (!runningStatus) {
                try {
                    Thread.sleep(1000); // Sleep for a short duration to avoid busy waiting
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt(); // Restore the interrupted status
                    break; // Exit the loop if the thread is interrupted
                }
            }
            boolean isDirEmpty = isModelDirectoryEmpty(com.param.modelPath);
            if (runningStatus && !isDirEmpty){
                System.out.println("Prepare is Finished.");
                if (cfg.isHeader()){
                    updateIsDirEmpty(isDirEmpty);
                }
                System.out.println("Should start the inference: "+shouldStartInference);
            }

            // 4. Starting from here we need to based on the ACTION_ENTER_CHAT_SCREEN to start inference
            if (cfg.isHeader()) {
                while (!shouldStartInference) {
                    try {
                        Thread.sleep(1000); // Sleep for a short duration to avoid busy waiting
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt(); // Restore the interrupted status
                        break; // Exit the loop if the thread is interrupted
                    }
                }
            }


            if (shouldStartInference && cfg.isHeader()){
                // 4.1 parameters set for classification task
                com.param.classes = new String[]{"Negative", "Positive"};
                // 4.2 Dataset would be used if we need conduct evaluation experiment
                Dataset dataset = null;

                while (com.param.numSample <= 0)
                    Thread.sleep(1000);

                // 4.3 Create input string array to store user input query. By default, the array size
                // is set to 1 for testing single-turn chat conversation.

                // 4.4 Based on whether user give input to run the inference
                ArrayList<String> test_input = new ArrayList<>();

                // 4.4.1 Receive userinput from chatscreen and save it to test_input array
                while (!messageStatus) {
                    try {
                        Thread.sleep(1000); // Sleep for a short duration to avoid busy waiting
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt(); // Restore the interrupted status
                        break; // Exit the loop if the thread is interrupted
                    }
                }

                // TODO: Need to fix the repetitive generation issue

                if (cfg.isHeader()) {
                    new Thread(() -> {
                        int j = 0;
                        String userinput = "";
                        while (j < com.param.numSample) {
                            if (messageContent.equals(userinput)){
                                try {
                                    Thread.sleep(1000);
                                } catch (InterruptedException e) {
                                    throw new RuntimeException(e);
                                }
                            }else {
                                System.out.println("New user input");
                                System.out.println("***************" + messageContent);
                                messageContent = String.format("User: %s. Response:", messageContent); // format chat prompt
                                userinput = messageContent;
                                test_input.add(userinput);
                                j++;
                            }
                        }
                    }).start();

                }

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
                Log.d(TAG, "Results Computation Time: " + (System.nanoTime() - startTime) / 1000000000.0);
                return null;
            }

            else if (!shouldStartInference && !cfg.isHeader()){ // running on devices are not header

                com.param.classes = new String[]{"Negative", "Positive"};
                Dataset dataset = null;
                while (com.param.numSample <= 0)
                    Thread.sleep(1000);
//                String[] test_input = new String[com.param.numSample];
                ArrayList<String> test_input = new ArrayList<>();
                int corePoolSize = 2;
                int maximumPoolSize = 2;
                int keepAliveTime = 500;

                try {
                    Log.d(TAG, "communication starts to running");
                    com.running(corePoolSize, maximumPoolSize, keepAliveTime, test_input);
                } catch (IOException | InterruptedException e) {
                    throw new RuntimeException(e);
                }
                results = com.timeUsage;
                return null;
            }
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
    public void onCreate() {
        super.onCreate();
        isServiceRunning = true;
        EventBus.getDefault().register(this);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        isServiceRunning = false;
        EventBus.getDefault().unregister(this);
    }

    public native int binaryClassify(byte[] data);

}
