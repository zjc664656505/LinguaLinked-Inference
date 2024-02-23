''''''
import json
import time
from threading import Thread

"""
    R: Ready
        client -> root; Show the status of edge device
    O: Open
        root -> client; The information of this task, e.g: Training/Inference/Task name, etc. 
    P: Prepare
        root -> client; Send the decentralized model and training/Inference code to clients. 
    I:  Initialized
        client -> root; Models are initialized and training/Inference is ready
    S: Start
        root -> client; Start training/Inference and Data transmission)    
    F: Finish
        client -> root; Finish training/Inference
    C: Close
        root -> client; Close the connection
"""

def communication_open_close(sender, config, status, conditions, lock, open=True):
    ## Status: Ready, Open, Prepare, Initialized, Start, Running, Finish
    while True:
        print('enter communication open close')
        with lock:
            info = sender.recv_multipart()
        client_id = info[0]
        msg = info[1]
        print(client_id + msg)
        ## Ready
        if open and msg == b'Ready':
            ## Open
            if len(info) != 3:
                print("Error")

            config["ids"][client_id] = info[2]
            print(config["ids"])

            status[client_id] = b'Ready'

            sender.send_multipart([client_id, b'Open',
                                    config["graph"],
                                    config["session_index"],
                                    config["task_type"],
                                    config["core_pool_size"],
                                    config["num_sample"],
                                    config["max_length"],
                                    json.dumps(config["dependency"]).encode()
                                   ])

            status[client_id] = b'Open'
            print(f"Status: Open {config['ids'][client_id]}")
            ## Prepare

            communication_prepare(sender, config, client_id, status)

            print(f"Status: Prepare {config['ids'][client_id]}")

        ## Initialized
        elif msg == b'Initialized':
            status[client_id] = b'Initialized'
            print(f"Status: Initialized {config['ids'][client_id]}")

            with conditions[1]:
                while not check_status(status, config, b"Initialized"):
                    conditions[1].wait()
                conditions[1].notify_all()

            ## Start
                sender.send_multipart([client_id, b"Start"])
                status[client_id] = b'Start'

                print(f"Status: Start {config['ids'][client_id]}")

        elif msg == b"Running":
            # Todo simulate load balance
            # time.sleep(10)
            # print(f"{config['ids'][client_id]} Start Load Balance")
            # # config["session_index"] = ";".join(["0,1", "2,3,4,5,6", "7,8,9"]).encode('utf-8')
            # sender.send_multipart([client_id, b"re-balance",
            #                                   config["session_index"],
            #                                   json.dumps(config["dependency"]).encode()])
            #
            # if (config["ids"][client_id] == config["head_node"].encode()):
            #     client_id, msg = sender.recv_multipart()
            #     config["reload_sampleId"] = msg.decode()
            #     print(f"The Reload Sample starts from {config['reload_sampleId']}")
            #     assert config["reload_sampleId"].isdigit(), f"reload sampleId is not an integer string"
            # else:
            #     while (config["reload_sampleId"] == None):
            #         print("Wait the resample ID")
            #         time.sleep(0.1)
            #
            #     print(f"Send Reload Sample id : {config['reload_sampleId']} to {config['ids'][client_id]}")
            #     sender.send_multipart([client_id, "id".encode(), config["reload_sampleId"].encode()])
            pass
        elif msg == b'Finish':
            status[client_id] = b'Close'
            with conditions[2]:
                while not check_status(status, config, b"Close"):
                    conditions[2].wait()
                conditions[2].notify_all()

                sender.send_multipart([client_id, b"Close"])
            print(f"Close {config['ids'][client_id]}")
            break

def send_model_file(path, sock, client_id, chunked=True, chunk_size=1024*1024):
    if not chunked:
        with open(path, 'rb') as f:
            data = f.read()
            sock.send_multipart([client_id, data])
            print("Data is sent")
    else:
        with open(path, 'rb') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    sock.send_multipart([client_id, b''])
                    break
                sock.send_multipart([client_id, chunk])
            print("Data is sent")

def communication_prepare(sender, config, client_id, status):
    sender.send_multipart([client_id, b'Prepare'])
    node_ip = config["ids"][client_id]
    sender.send_multipart([client_id, str(config["skip_model_transmission"]).encode()])

    if not config["skip_model_transmission"]:   ## Assume data is received on the machine
        print(f"send {config['file_path'][node_ip]} to {node_ip}")

        # onnx sends multiple files, which should be a zip
        if config["onnx"]:
            send_model_file(config["file_path"][node_ip], sender, client_id)
        else:
            send_model_file(config["file_path"][node_ip], sender, client_id)

        # # transmit tokenizer to header
        # if config["head_node"] == node_ip:
        #     print(f"send {config['file_path'][b'tokenizer']} to {node_ip}")
        #     send_model_file(config["file_path"][b"tokenizer"], sender, client_id)

    status[client_id] = b"Prepare"

def communication_data_transmission(sender, num_devices, head_client_id, status):
    while check_status(status, num_devices, b"Start"):
        pass


def communication_result_transmission(sender, result, num_devices, tail_client_id, status):
    # while check_status(status, num_devices, b"Finish"):
    sender.send_multipart([b"res", result])
    pass


all_status = {b"Ready":  0,
              b"Open":   1,
              b"Prepare":2,
              b"Initialized": 3,
              b"Start":  4,
              b"Running":5,
              b"Finish": 6,
              b"Close": 7}

def check_status(status, config, mode):
    if len(status) != config["num_device"]:
        return False
    for v in status.values():
        if all_status[v] < all_status[mode]:
            return False
    return True


def ConfigCreator(Config, client_id):
    ## Based on the monitor situation
    return Config["graph"]