package com.example.SecureConnection;

import org.json.JSONException;
import org.zeromq.ZMQ;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.Semaphore;
import org.zeromq.ZMQ.Socket;

public class EndProcess implements Runnable{
    private Config cfg;
    private Communication com;
    private final Semaphore latch;
    private final Map<Integer, ZMQ.Socket> serverSocket;
    private final Map<Integer, ZMQ.Socket>  clientSocket;

    public EndProcess(Config cfg, Communication com, Semaphore latch) {
        this.com = com;
        this.cfg = cfg;
        ArrayList<Map<Integer, Socket>> sockets = null;
        try {
            sockets = com.allSockets.take();
        } catch (InterruptedException e) {
            System.out.println("Waiting for an element from the sockets queue...");
            e.printStackTrace();
        }
        clientSocket = sockets.get(0);
        serverSocket = sockets.get(1);

        this.latch = latch;
    }

    public int obtainResultsFromTailer(Socket serverSocket, int receivedId) {
        // Special for header to obtain results from tailer
        if (cfg.isHeader()) {
//                System.out.println("Start to be a Client");
            // Handle the case header to request tailer results
            receivedId = Utils.convertByteArrayToInt(serverSocket.recv(0));
            byte[] res = serverSocket.recv(0);
            if (com.param.task_type.equals("generation")) {
                int decode_id = Utils.convertByteArrayToInt(res);
                com.InputIds.get(receivedId).add(decode_id);
            } else {
                com.logits.put(receivedId, res);
            }
            System.out.println("No." + receivedId + " Results Obtained");
        }
        return receivedId;
    }


    public int finishSteps(int receivedId,
                           Map<Integer, Socket> serverSocket,
                           Map<Integer, Socket>  clientSocket) throws InterruptedException, JSONException, JSONException {
        long startTime = System.nanoTime();

        Communication.OneStep step = com.new OneStep(receivedId, serverSocket, clientSocket);

        receivedId = step.procssingAsClient(receivedId);
        System.out.println("No." + receivedId + " Part1 End Process Time: " + (System.nanoTime() - startTime) / 1000000000.0);

        if (!cfg.isHeader()) {
            startTime = System.nanoTime();
            com.inferenceProcedure(receivedId);
            System.out.println("No." + receivedId + " Part2 End Process Time: " + (System.nanoTime() - startTime) / 1000000000.0);

            startTime = System.nanoTime();
            step.processAsServer(receivedId);
            System.out.println("No." + receivedId + " Part3 End Process Time: " + (System.nanoTime() - startTime) / 1000000000.0);
        }

        startTime = System.nanoTime();
        receivedId = step.obtainResultsFromTailer(receivedId);
        System.out.println("No." + receivedId + " Part4 End Process Time: " + (System.nanoTime() - startTime) / 1000000000.0);

        return receivedId;
    }

    @Override
    public void run() {
        System.out.println("++++++++++++ End Processing ++++++++++ ");

        int receivedId = com.param.numSample - 1;

        if (com.param.max_length == 0) {
            // classification
            System.out.println("++++++++++++SampleID: " + receivedId);
            try {
                receivedId = finishSteps(receivedId, serverSocket, clientSocket);
            } catch (InterruptedException | JSONException e) {
                throw new RuntimeException(e);
            }
        } else {
            // generation
            for (int m = 0; m < com.param.max_length; m++) {
                long startTime = System.nanoTime();
                System.out.println("++++++++++++SampleID: " + receivedId + "++++++++++TokenID:" + m);
                try {
                    receivedId = finishSteps(receivedId, serverSocket, clientSocket);
                } catch (InterruptedException | JSONException e) {
                    throw new RuntimeException(e);
                }
                System.out.println("Token Process Time: " + (System.nanoTime() - startTime) / 1000000000.0);
            }
        }

        try {
            com.allSockets.put(new ArrayList<Map<Integer, Socket>>(){{
                add(clientSocket);
                add(serverSocket);
            }});
//                serverSockets.put(serverSocket);
//                clientSockets.put(serverSocket);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        com.cleanUpBuffer(receivedId);
        latch.release();

    }


}