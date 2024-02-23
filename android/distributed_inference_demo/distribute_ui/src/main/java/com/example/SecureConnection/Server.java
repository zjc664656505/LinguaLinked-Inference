package com.example.SecureConnection;

import org.zeromq.SocketType;
import org.zeromq.ZMQ.Socket;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;
import org.zeromq.ZMQException;

import java.util.Map;
import java.util.concurrent.CountDownLatch;

public class  Server {
    /***
     On behalf of the server, communicate with the client devices.
     ***/

    public Map<String, Socket> nextNodes;

    public Server() {}

    public Socket establish_connection(ZContext context, SocketType type, int port) {
        Socket socket = context.createSocket(type);
        socket.bind("tcp://*:" + port);
        socket.setIdentity(Config.local.getBytes());
        return socket;
    }


//    class CommunicationAsServer implements Runnable {
//        private String targetIP;
//        private Socket sender;
//        private final CountDownLatch countDown;
//
//        CommunicationAsServer(String targetIp, Socket sender, CountDownLatch countDown) {
//            this.targetIP = targetIP;
//            this.sender = sender;
//            this.countDown = countDown;
//        }
//
//        @Override
//        public void run() {
//            try {
////                String msg = new String(sender.recv(0));
//                ZMsg receivedMsg = ZMsg.recvMsg(sender);
//                // Get the identity frame (client ID)
//                ZFrame identity = receivedMsg.unwrap();
//                String msg = receivedMsg.getLast().toString();
//
//                if (msg.contains("Request data")) {
//                    System.out.println(param.choice);
//                    ZMsg replyMsg = new ZMsg();
//                    replyMsg.wrap(identity.duplicate());
//                    countDown.await();
//                    replyMsg.add(OutputData.get(param.choice));
//                    replyMsg.send(sender);
//                }
//            } catch (InterruptedException e) {
//                throw new RuntimeException(e);
//            }
//
//        }
//    }
}