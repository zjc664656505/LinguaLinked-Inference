package com.example.distribute_ui;

public class Events {
    // Event for registering the communication running status for service communication
    public static class RunningStatusEvent{
        public final boolean isRunning;
        public RunningStatusEvent(boolean isRunning){
            this.isRunning = isRunning;
        }
    }

    // Event for UI-service communication to let the background service know the inference chat can initiate
    public static class messageSentEvent{
        public final boolean messageSent;
        public final String messageContent;
        public messageSentEvent(boolean messageSent, String messageContent){
            this.messageSent = messageSent;
            this.messageContent = messageContent;
        }
    }

    public static class enterChatEvent{
        public final boolean enterChat;
        public enterChatEvent(boolean enterChat){
            this.enterChat = enterChat;
        }
    }

    public static class sampleIdEvent{
        public final int sampleId;
        public sampleIdEvent(int sampleId){
            this.sampleId = sampleId;
        }
    }
}
