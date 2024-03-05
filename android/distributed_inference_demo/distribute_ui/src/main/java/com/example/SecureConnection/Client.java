package com.example.SecureConnection;

import static com.example.distribute_ui.BackgroundService.TAG;

import android.content.Intent;
import android.util.Log;

import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.TimeUnit;
import java.io.File;
import java.io.FileOutputStream;
import java.util.zip.ZipOutputStream;

import org.greenrobot.eventbus.EventBus;
import org.json.JSONException;
import org.json.JSONObject;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;
import org.zeromq.ZMQ.Socket;

import java.io.IOException;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipEntry;
import com.example.SecureConnection.Utils;
import com.example.distribute_ui.BackgroundService;
import com.example.distribute_ui.Events;

public class Client {

    public Socket establish_connection(ZContext context, SocketType type, int port, String address) {
        Socket socket = context.createSocket(type);
        socket.connect("tcp://" + address + ":" + port);
        return socket;
    }

    public void communicationOpenClose(Config cfg, Communication com, Socket receiver) throws Exception {
        Communication.Params param = com.param;
        while (true) {
            // Ready
            if (param.status.equals("Ready")) {
                System.out.println("Status: Ready");
                Log.d(TAG, "Status: Ready");
                receiver.sendMore("Ready");

                receiver.send(Config.local, 0);     // Send device IP to Server in Ready Status
                Log.d(TAG, "waiting for open signal");

                // Open
                String msg = new String(receiver.recv(0));
                Log.d(TAG, "msg: " + msg);
                if (msg.equals("Open")) {
                    param.status = "Open";
                    System.out.println("Status: Open");

                    receiveIPGraph(cfg, receiver);

                    receiveSessionIndex(receiver);

                    receiveTaskType(param,receiver);

                    receiveThreadPoolSize(param, receiver);

                    receiveBatchSize(param, receiver);

                    receiveSeqLength(param,receiver);

                    receiveDependencyMap(receiver);

                    Log.d(TAG, "open status receive info finished");
                }

                // Prepare
                msg = new String(receiver.recv(0));
                Log.d(TAG, "prepare msg: " + msg);
                if (msg.equals("Prepare")) {
                    communicationPrepare(receiver, param);
                }

                // Initialize
                LoadBalanceInitialization();
                modelInitialization(cfg, param);
                param.status = "Initialized";
                System.out.println("Status: Initialized");
                Log.d(TAG, "Status: Initialized");
                receiver.send("Initialized", 0);

                msg = new String(receiver.recv(0));
                System.out.println(msg);

                if (msg.equals("Start")) {
                    param.status = "Start";
                    System.out.println("Status: Start");
                    Log.d(TAG, "Status: Start");
                    receiver.send("Running");
                    System.out.println("Status: Running");
                    Log.d(TAG, "Status: Running");
                    param.status = "Running";
                    if (param.status == "Running") {
                        // post running events for letting background service know the running status is true
                        EventBus.getDefault().post(new Events.RunningStatusEvent(true));
                    }

//                    ZMQ.Poller poller = com.context.createPoller(1);
//                    poller.register(receiver, ZMQ.Poller.POLLIN);

                    // Waiting Load balance
//                    while (param.status.equals("Running")) {
//                        int poll = poller.poll(500); // Check every 0.5s
//                        if (poll > 0) {
//                            String signal = receiver.recvStr(0);
//                            System.out.println("===================== Received: " + signal);
//                            Log.d(TAG, "===================== Received: " + signal);
//                            if (signal.equals("re-balance")) {
//                                receiveSessionIndex(receiver);
//                                receiveDependencyMap(receiver);
//                                if (cfg.isHeader) {
//                                    Communication.LB_Pause.setConditionTrue();
//                                    Communication.loadBalance.setReSampleId(com.sampleId+1);
//                                    receiver.send(Integer.toString(com.sampleId+1));
//                                } else {
//                                    Communication.LB_Pause.setConditionTrue();
//                                    String id = receiver.recvStr(0);
//                                    String id_Str = receiver.recvStr(0);
//                                    int resampleId = Integer.parseInt(id_Str);
//                                    System.out.println("===================== Received resampleId: " + resampleId);
//                                    Log.d(TAG, "===================== Received resampleId: " + resampleId);
//                                    Communication.loadBalance.setReSampleId(resampleId);
//                                }
//                            }
//                        }
//                    }
//                    break;
                }

            } else if (param.status.equals("Finish")) {
                receiver.send("Finish");
                String msg = new String(receiver.recv(0));
                System.out.println(msg);
                System.out.println("Status: Close");
                Log.d(TAG, "Status: Close");

                if (msg.equals("Close")) {
                    for(ArrayList<Map<Integer, Socket>> s: com.allSockets)
                        closeSockets(s);
//                    com.context.close();
                }
                System.out.println("Finish task");
                Log.d(TAG, "Finish task");
                break;
            }
        }
    }
    public void receiveModelFile(String path, Socket receiver, boolean chunked, int chunk_size) {

        File file = new File(path);
        if (file.exists() && file.delete()) {
            System.out.println("Deleted the file: " + file.getName());
        } else {
            System.out.println("Failed to delete the file.");
        }

        File parentDir = file.getParentFile();
        System.out.println("parent dir is: " + parentDir.toString());
        assert parentDir != null;
        if (!parentDir.exists()) {
            parentDir.mkdirs();
        }
        System.out.println("Start receiving file");
        Log.d(TAG, "Start receiving file");

        file = new File(path);
        if (!chunked) {
            // Directly write the entire file
            try (FileOutputStream fos = new FileOutputStream(file)) {
                byte[] data = receiver.recv(0);
                fos.write(data);
                System.out.println("Data is written");
                Log.d(TAG, "Data is written");
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            // Receive byte file
            try (FileOutputStream fos = new FileOutputStream(file)) {
                byte[] chunk;
                int totalSize = 0;
                while ((chunk = receiver.recv()) != null) {
                    fos.write(chunk);
                    totalSize += chunk.length;
                    if (chunk.length == 0) {
                        break;
                    }
                    System.out.println("Chunk size: " + chunk.length + " Total size: " + totalSize);
                    Log.d(TAG, "Chunk size: " + chunk.length + " Total size: " + totalSize);
                }
                System.out.println("Data is written");
                Log.d(TAG, "Data is written");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void LoadBalanceInitialization() throws Exception {
        Communication.loadBalance.reLoadBalance();
        System.out.println("load balance init finished");
        Log.d(TAG, "load balance init finished");
    }

    public void communicationPrepare(Socket receiver, Communication.Params param) {
        param.status = "Prepare";
        boolean chunk = true;
        System.out.println("Status: Prepare");
        Log.d(TAG, "Status: Prepare");
        String skipModelDownload = new String(receiver.recv(0));
        if (skipModelDownload.equals("False")) {  // Skip the model, causing model exists
            receiveModelFile(param.modelPath + "/module.zip", receiver, chunk, 10 * 1024 * 1024);  // chunked 1MB
            System.out.println("Model Received");
            Log.d(TAG, "Model Received");
//            if (cfg.isHeader) {
//                Log.d(TAG, "start receiving tokenizer");
//                receiveModelFile(param.modelPath + "/device/tokenizer.json", receiver, chunk, 1024 * 1024);  // chunked 1MB
//                System.out.println("Tokenizer Received");
//                Log.d(TAG, "Tokenizer Received");
//            }

        } else {
            System.out.println("Model Exists");
            Log.d(TAG, "Model Exists");
        }
        if (skipModelDownload.equals("False")){
            Utils.unzipFile(param.modelPath + "/module.zip");
        }

    }

    public void modelInitialization(Config cfg, Communication.Params param) {
//        for (String i: Communication.sessionIndex) {
////            Communication.sessions.add(createSession(param.modelPath + "/device/module" + i + "/module_" + i + ".onnx"));
//            Communication.sessions.add(createSession(param.modelPath + "/device/module.onnx"));
//            System.out.println("Load module " + i + " successfully");
//            Log.d(TAG, "Load module " + i + " successfully");
//        }
        Communication.sessions.add(createSession(param.modelPath + "/device/module.onnx"));
        System.out.println("create session finished");

        if (cfg.isHeader() || cfg.isTailer()) {
            Communication.tokenizer = createHuggingFaceTokenizer(param.modelPath + "/device/tokenizer.json");
            // OR SENTENCEPIECE LATER
            System.out.println("Tokenizer created");
            Log.d(TAG, "Tokenizer created");
        }
        System.out.println("model init finished");
    }

    private void receiveIPGraph(Config cfg, Socket receiver){
        // Receive IP Graph
        byte[] ip_graph = receiver.recv(0);
        String ip_graph_str = new String(ip_graph);
        cfg.buildCommunicationGraph(ip_graph_str);
//        System.out.println("Get IP graph");
        Log.d(TAG, "Get IP graph: " + ip_graph_str);
        cfg.getDeviceId();
    }

    private void receiveSessionIndex(Socket receiver){
        // Receive Session Index and inital load balance
        String session_indices = receiver.recvStr(0);
        Communication.loadBalance.sessIndices = session_indices.split(";");
        Log.d(TAG, "Get session index: " + session_indices);
    }

    private void receiveTaskType(Communication.Params param, Socket receiver){
        //  Receive Task Type
        byte[] task_type = receiver.recv(0);
        param.task_type = new String(task_type);
//        System.out.println("Task: " + param.task_type);
        Log.d(TAG, "Task: " + param.task_type);
        if (param.task_type.equals("generation")) {
//            System.out.println("Generation with text length: " + param.max_length);
            Log.d(TAG, "Generation with text length: " + param.max_length);
        }else if (param.task_type.equals("classification")){
//            System.out.println("Classification without text length");
            Log.d(TAG, "Classification without text length");
        }
    }

    private void receiveThreadPoolSize(Communication.Params param, Socket receiver){
        //  Receive thread pool size
        String pool_size = "";
        try {
            byte[] core_pool_size = receiver.recv(0);
            pool_size = new String(core_pool_size);
            param.corePoolSize = Integer.parseInt(pool_size);
        } catch (NumberFormatException nfe) {
            System.out.println("Core Pool Size is not Integer");
        }
        Log.d(TAG, "Get ThreadPollSize: " + pool_size);
    }

    private void receiveBatchSize(Communication.Params param, Socket receiver){
        //  Receive batch size
        try {
            byte[] batch = receiver.recv(0);
            param.numSample = Integer.parseInt(new String(batch));
        } catch (NumberFormatException nfe) {
            System.out.println("Num of Batch is not Integer");
        }
//        System.out.println("Num of Sample: " + param.numSample);
        Log.d(TAG, "Num of Sample: " + param.numSample);
    }

    private void receiveSeqLength(Communication.Params param, Socket receiver) {
        // Receive Max length for generation task, 0 for classification task
        try {
            byte[] max_length = receiver.recv(0);
            param.max_length = Integer.parseInt(new String(max_length));
        } catch (NumberFormatException nfe) {
//            System.out.println("max_length is not Integer");
            Log.d(TAG, "max_length is not Integer");
        }
        Log.d(TAG, "Get Sequence Length: " + param.max_length);
    }

    private void receiveDependencyMap(Socket receiver) {
        String depMap = receiver.recvStr(0);
//        System.out.println("Show Map: " + depMap);
        Log.d(TAG, "Show Map: " + depMap);
        try {
            Communication.loadBalance.dependencyMap = new JSONObject(depMap);
        }catch (JSONException e) {
//            System.out.println("JSON EXCEPTION");
            Log.d(TAG, "Dependency Map JSON EXCEPTION");
        }
        Log.d(TAG, "Get Dependency Map");
    }

    public void closeSockets(ArrayList<Map<Integer, Socket>> sockets) {
//        releaseSession(Communication.session);
        for (Map<Integer, Socket> sock: sockets) {
            for (Socket socket : sock.values()) {
                socket.close();
            }
        }
    }
    public static native long createSession(String inference_model_path);

    public static native long releaseSession(long session);
    public native long createHuggingFaceTokenizer(String tokenizer_path);

    public native long createSentencePieceTokenizer(String tokenizer_path);
}