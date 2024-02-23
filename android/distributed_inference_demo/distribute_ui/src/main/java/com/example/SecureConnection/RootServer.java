//package com.example.SecureConnection;
///*
//    R: Ready
//        client -> root; Show the status of edge device
//    O: Open
//        root -> client; The information of this task, e.g: Training/Inference/Task name, etc.
//    P: Prepare
//        root -> client; Send the decentralized model and training/Inference code to clients.
//    I:  Initialized
//        client -> root; Models are initialized and training/Inference is ready
//    S: Start
//        root -> client; Start training/Inference and Data transmission)
//    F: Finish
//        client -> root; Finish training/Inference
//    C: Close
//        root -> client; Close the connection
// */
//
//import org.zeromq.ZMQ;
//import org.zeromq.ZMQ.Socket;
//
//
//import java.util.Map;
//import java.util.concurrent.locks.Lock;
//
//public class RootServer extends Server implements Runnable{
//    private Lock lock;
//    private Socket sender;
//    private Map<String, String> status;
//
//    RootServer(Socket sender){
////        this.lock = new Lock();
//        this.sender = sender;
//    }
//
//    @Override
//    public void run() {
//        while (true){
//            lock.lock();
//            String id = new String(sender.recv());
//            String msg = new String(sender.recv());
//            lock.unlock();
//            if (msg.equals("Ready")){
//
//            }
//        }
//
//    }
//}
//
//
////import org.zeromq.ZContext;
////        import org.zeromq.ZMQ;
////        import org.zeromq.ZMsg;
////
////public class Server {
////    public static void main(String[] args) {
////        try (ZContext context = new ZContext()) {
////            // Create and bind the ROUTER socket
////            ZMQ.Socket router = context.createSocket(ZMQ.ROUTER);
////            router.bind("tcp://*:5555");
////
////            while (!Thread.currentThread().isInterrupted()) {
////                // Receive a message from the router socket
////                ZMsg receivedMsg = ZMsg.recvMsg(router);
////
////                // Get the identity frame (client ID)
////                ZFrame identity = receivedMsg.unwrap();
////
////                // Get the content of the message
////                String content = receivedMsg.getLast().toString();
////
////                System.out.println("Received message from client " + identity + ": " + content);
////
////                // Process the received message
////
////                // Create a new message to send to clients
////                ZMsg replyMsg = new ZMsg();
////
////                // Set the identity frame (client ID)
////                replyMsg.wrap(identity.duplicate());
////
////                // Set the content of the message
////                replyMsg.add("Server reply");
////
////                // Send the message to the router socket
////                replyMsg.send(router);
////            }
////        }
////    }
////}
//
//
////import org.zeromq.ZContext;
////        import org.zeromq.ZMQ;
////        import org.zeromq.ZMsg;
////
////public class Client {
////    public static void main(String[] args) {
////        try (ZContext context = new ZContext()) {
////            // Create and connect the DEALER socket
////            ZMQ.Socket dealer = context.createSocket(ZMQ.DEALER);
////            dealer.connect("tcp://localhost:5555");
////
////            // Set the client ID as a unique string
////            String clientId = "client-" + Thread.currentThread().getId();
////
////            while (!Thread.currentThread().isInterrupted()) {
////                // Create a new message
////                ZMsg msg = new ZMsg();
////
////                // Add the client ID as the first frame
////                msg.add(clientId);
////
////                // Add the content of the message
////                msg.add("Hello from client");
////
////                // Send the message to the dealer socket
////                msg.send(dealer);
////
////                // Wait for the server's reply
////                ZMsg replyMsg = ZMsg.recvMsg(dealer);
////
////                // Get the content of the reply
////                String replyContent = replyMsg.getLast().toString();
////
////                System.out.println("Received reply from server: " + replyContent);
////
////                // Process the reply message
////            }
////        }
////    }
////}
