package com.example.SecureConnection;

import android.util.Log;

import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;

// Configuration for the communication system
public class Config {
    public static final String TAG = "ConfigCreator";

    public static String local;

    public static int port;

    public String root;
    public int rootPort;
    public List<String> prevNodes;
    public List<String> nextNodes;
    public  boolean isHeader;
    public  boolean isTailer;
    public  boolean isDecoder;
    public int deviceId;

    public int k;

    public float initial_temp;

    public float final_temp;
    public String[] ipGraph;

    public Config(String root, int rootPort){
        Config.local = getCurrentDeviceIP();
        Config.port = 12345;
        this.root = root;
        this.rootPort = rootPort;
        this.isHeader = false;
        this.isTailer = false;
    }
    public Config(String root, int rootPort, int k, float initial_temp){
        Config.local = getCurrentDeviceIP();
        Config.port = 12345;
        this.root = root;
        this.rootPort = rootPort;
        this.isHeader = false;
        this.isTailer = false;
        this.k = k;
        this.initial_temp = initial_temp;
    }

    public Config(String root, int rootPort, List<String> prevNodes, List<String> nextNodes, boolean header, boolean tailer){
        this.root = root;
        this.rootPort = rootPort;
        this.nextNodes = nextNodes;
        this.prevNodes = prevNodes;
        this.isHeader = header;
        this.isTailer = tailer;
    }

    private String getCurrentDeviceIP() {
        try {
            for (Enumeration<NetworkInterface> en = NetworkInterface.getNetworkInterfaces(); en.hasMoreElements();) {
                NetworkInterface networkInterface = en.nextElement();
                for (Enumeration<InetAddress> enumIpAddr = networkInterface.getInetAddresses(); enumIpAddr.hasMoreElements();) {
                    InetAddress inetAddress = enumIpAddr.nextElement();
                    if (!inetAddress.isLoopbackAddress() && inetAddress instanceof Inet4Address) {
                        return inetAddress.getHostAddress().toString();
                    }
                }
            }
        } catch (SocketException ex) {
            Log.e(TAG, String.format("Current Device IP %s", ex.toString()));
        }
        return null;
    }

    public void setNextNodes(List<String> nextNodes) {
        this.nextNodes = nextNodes;
    }

    public void setPrevNodes(List<String> prevNodes) {
        this.prevNodes = prevNodes;
    }

    public void setHeader(boolean header) {
        this.isHeader = header;
    }

    public void setTailer(boolean Tailer) {
        this.isTailer = Tailer;
    }

    public boolean isTailer(){
        return isTailer;
    }
    public boolean isHeader() {
        return isHeader;
    }


    public boolean isDecoderHeader(){
        return isDecoder;
    }

    public void buildCommunicationGraph(String graph){
        this.ipGraph =  graph.split(",");

        if (ipCheck(ipGraph)){
            if (local.equals(ipGraph[0])) {
                setHeader(true);
                setPrevNodes(new ArrayList<>(Arrays.asList(ipGraph[ipGraph.length-1])));
                setNextNodes(new ArrayList<>(Arrays.asList(ipGraph[1])));
            }else if (local.equals(ipGraph[ipGraph.length-1])) {
                setTailer(true);
                setPrevNodes(new ArrayList<>(Arrays.asList(ipGraph[ipGraph.length-2])));
                setNextNodes(new ArrayList<>(Arrays.asList(ipGraph[0])));
            }else {
                for (int i = 1; i < ipGraph.length-1; i++){
                    if (local.equals(ipGraph[i])){
                        setPrevNodes(new ArrayList<>(Arrays.asList(ipGraph[i-1])));
                        setNextNodes(new ArrayList<>(Arrays.asList(ipGraph[i+1])));
                    }
                }
            }
        }else{
            System.out.println("Receive Bad IP Graph");
        }
    }

    public void getDeviceId() {
        if (ipGraph != null) {
            deviceId = Arrays.asList(ipGraph).indexOf(Config.local);
        }else {
            deviceId = -1;
        }
    }

    public int prevDeviceId() {
        if (deviceId != -1) {
            return Arrays.asList(ipGraph).indexOf(prevNodes.get(0));
        }else{
            return -1;
        }
    }

    public int nextDeviceId() {
        if (deviceId != -1) {
            return Arrays.asList(ipGraph).indexOf(nextNodes.get(0));
        }else{
            return -1;
        }
    }

    public boolean ipCheck(String[] ipGraph){
        //Todo
        return true;
    }
}