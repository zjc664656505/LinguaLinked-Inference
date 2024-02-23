package com.example.SecureConnection;

import android.os.AsyncTask;

import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

import java.nio.charset.StandardCharsets;
public class Test {
    private static final String PING = "ping";
    private static final String PONG = "pong";
    public void connectToServer() throws InterruptedException {
        try (ZContext context = new ZContext()) {
            // Start a router in a new thread
            new Thread(() -> {
                ZMQ.Socket router = context.createSocket(SocketType.ROUTER);
                router.bind("tcp://*:12345");

                while (!Thread.currentThread().isInterrupted()) {
                    // Receive a message
                    System.out.println("start...");
                    byte[] identity = router.recv(0);
                    byte[] msg = router.recv(0);
                    System.out.println(new String(msg));

                    // If we received a ping, reply with a pong
                    if (PING.equals(new String(msg))) {
                        router.sendMore(identity);
                        router.send(PONG);
                    }
                    System.out.println("Pong...");
                }
            }).start();
        }
    }
}

//
//public class Test {
//    private static final String SERVER_IP = "192.168.0.45";
//    private static final int SERVER_PORT = 5555;
//
//    public void connectToServer() {
//            ZContext context = new ZContext();
//            ZMQ.Socket socket = context.createSocket(SocketType.REQ);
//
//            try {
//                // Connect to the server
//                String serverAddress = "tcp://" + SERVER_IP + ":" + SERVER_PORT;
//                socket.connect(serverAddress);
//                System.out.println("Connected to server: " + serverAddress);
//
//                // Send a message to the server
//                String message = "Hello, Server!";
//                socket.send(message.getBytes(), 0);
//                System.out.println("Sent message: " + message);
//
//                // Receive a response from the server
//                byte[] reply = socket.recv(0);
//                String response = new String(reply);
//                System.out.println("Received response: " + response);
//
//            } catch (Exception e) {
//                e.printStackTrace();
//            } finally {
//                // Close the socket and terminate the context
//                socket.close();
//                context.destroy();
//            }
//        }
//    }
//
