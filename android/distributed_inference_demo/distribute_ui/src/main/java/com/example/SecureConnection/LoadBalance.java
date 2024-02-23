package com.example.SecureConnection;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class LoadBalance {
    public Communication com;
    public Config cfg;
    public int reSampleId;
    public String[] sessIndices;
    public JSONObject dependencyMap; //sender_seq, sender_res, receiver_seq, receiver_res
    public LoadBalance(Communication com, Config cfg) {
        this.com = com;
        this.cfg = cfg;
        this.reSampleId = -1;
    }
    public void setReSampleId(int id){
        this.reSampleId = id;
    }
    public void reLoadBalance() throws Exception {

        assert sessIndices != null;
        updateModule_On_Devices();

        assert dependencyMap != null;
        JSONObject sender_seq = dependencyMap.getJSONObject("send_seq");
        JSONObject sender_res = dependencyMap.getJSONObject("send_res");
        JSONObject receiver_seq = dependencyMap.getJSONObject("rece_seq");
        JSONObject receiver_res = dependencyMap.getJSONObject("rece_res");

        com.sendDeviceIndex = new HashSet<>();
        com.receiveDeviceIndex = new HashSet<>();
        com.sendIndex = new TreeMap<>();
        com.receiveIndex = new TreeMap<>();

        // 0: [seq JSON object, res JSON object]
        // 1: [seq JSON object, res JSON object]
        // 2: ......


        for (String i: Communication.sessionIndex) {

            com.sendIndex.put(i, new ArrayList<JSONObject>(){{
                if (sender_seq.has(i))
                    add(sender_seq.getJSONObject(i));

                if (sender_res.has(i)) {
                    JSONObject obj =  sender_res.getJSONObject(i);
                    add(obj);
                    // add all res devices
                    Iterator<String> keys = obj.keys();
                    while (keys.hasNext())
                        com.sendDeviceIndex.add(com.module_on_devices.get(keys.next()));
                }

            }});
            // Add next device
            if (!cfg.isTailer)
                com.sendDeviceIndex.add(cfg.nextDeviceId());

            com.receiveIndex.put(i, new ArrayList<JSONObject>(){{
                if (receiver_seq.has(i))
                    add(receiver_seq.getJSONObject(i));
                if (receiver_res.has(i)){
                    JSONObject obj =  receiver_res.getJSONObject(i);
                    add(obj);
                    // add prev res devices
                    Iterator<String> keys = obj.keys();
                    while (keys.hasNext())
                        com.receiveDeviceIndex.add(com.module_on_devices.get(keys.next()));
                }}}
            );
            // Add prev res device
            if (!cfg.isHeader)
                com.receiveDeviceIndex.add(cfg.prevDeviceId());
        }


        com.sendDeviceIndex.remove(cfg.deviceId);
        com.receiveDeviceIndex.remove(cfg.deviceId);

        System.out.println("Device Send Set: "+ com.sendDeviceIndex);
        System.out.println("Device receive Set: "+ com.receiveDeviceIndex);

        Utils.sortElements(com.sendIndex);
        Utils.sortElements(com.receiveIndex);

        if (com.sendIndex.size() > 0)
            for (Map.Entry<String, ArrayList<JSONObject>> e: com.sendIndex.entrySet())
                if (e.getValue().size() > 1)
                    System.out.println("Module "+e.getKey() +" send res: "+  e.getValue().get(1).length());

        if (com.receiveIndex.size() > 0)
            for (Map.Entry<String, ArrayList<JSONObject>> e: com.receiveIndex.entrySet())
                if (e.getValue().size() > 1)
                    System.out.println("Module "+e.getKey() +" receive res: "+  e.getValue().get(1).length());

        com.getReceiveResDevice2Device();
        com.getSendResDevice2Device();
    }

    public void updateModule_On_Devices(){
        for(int i = 0; i < sessIndices.length; i++) {
            String [] sessionIdx = sessIndices[i].split(",");
            if (i == cfg.deviceId)
                Communication.sessionIndex = sessionIdx;
            for(String session: sessionIdx)
                com.module_on_devices.put(session, i);
        }
    }


    public void ModifySession() {
        List<String> new_idx = Arrays.asList(sessIndices[cfg.deviceId].split(","));
        TreeMap<String, Long> tmp = new TreeMap<>();

        for (int i = 0; i < Communication.sessionIndex.length; i++){
            tmp.put(Communication.sessionIndex[i], Communication.sessions.get(i));
            if (!new_idx.contains(Communication.sessionIndex[i])) {
                //Delete Session
                Long sess = tmp.remove(Communication.sessionIndex[i]);
                Client.releaseSession(sess);
                System.out.println("Session "+ Communication.sessionIndex[i]+ " Removed");
            }
        }
        for (String s : new_idx) {
            if (!tmp.containsKey(s)){
                // Add Session
                tmp.put(s, Client.createSession(com.param.modelPath + "/device/module_" +  s + ".onnx"));
                System.out.println("Session "+ s+ " Added");
            }
        }

        Communication.sessions = new ArrayList<>();
        Communication.sessions.addAll(tmp.values());
        System.out.println("Session Reload");
    }

}