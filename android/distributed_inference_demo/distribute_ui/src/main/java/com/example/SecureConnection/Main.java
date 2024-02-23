//package com.example.SecureConnection;
//
//import java.io.IOException;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.Collections;
//import java.util.List;
//
//public class Main {
//
//    public static void main(String[] args) throws IOException, InterruptedException {
//        Config cfg = new Config();
//        Config.local = "192.168.0.51";
//        Config.port = 12345;
//
//        cfg.root = "192.168.0.45";
//        cfg.rootPort = 23456;
//
//        cfg.nextNodes = new ArrayList<String>(){{
//                add("192.168.0.22");
//            }};
////        cfg.nextNodes = new ArrayList<>();
////        cfg.prevNodes = new ArrayList<>();
//        cfg.header = "192.168.0.51";
//        cfg.tailer = "192.168.0.51";
//
//        Communication com = new Communication(cfg);
//        com.param.modelPath = System.getProperty("user.dir") + "/src/main/resources/module.onnx";
////        com.prepare();
////        com.running();
//    }
//
//}