Decentralized Connection

Put the latest root.py and root_server.py into secureconnection python version

1. On Root Server side:
   + Modify the configuration, such as file cfg, skip_model_transmission in the Root.py
   + Please give the correct num_batch every time.
   + Run Root.py file 

2. On Edge Device side:
   + Open BackgroundSevice, give/uncomment the correct module config on correct device, 
   + Give the correct input_text as shown, num of input text matches num_batch
   + Run the MainActivity and install the Test1 on correct device.
   + (The model and tokenizer can be written as long as RAM is enough, set skip_model_transmission = False
   + Otherwise, place the model and tokenizer on the param.modelPath. /data/user/0/com.example.test1/files/module.onnx)
   
3. Runing decentralized inference
   + When all devices are installed
   + Run Root.py
   + Run all Test1 on all devices
   + The print out results can be seen in logcat, with the head device connected on android studio.
