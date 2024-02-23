import zmq
import argparse
import server
import client
import time
import yaml
import pickle
import torch
import threading
import torch.nn.functional as F
import os
from test_inference_resnet import inference
from test_inference_resnet_onnx import inference_onnx
from transformers import AutoTokenizer

def inference_procedure(cfg, received_data, param):
    while True:
        if cfg['local'] in received_data and received_data[cfg['local']] != None and param["model"] != None:
            model = param["model"]
            inference_data = pickle.loads(received_data[cfg['local']])
            print(f"The inference data is: {inference_data}")
            if param['onnx']:
                res = inference_onnx(inference_data, model, param["model_input_name"])
            else:
                res = inference(inference_data, model)
            received_data[cfg['local']] = pickle.dumps(res)
            param['choice'] = cfg['local']
            print("Inference is complete!")
            break

# Processing incoming data
def processing(cfg, received_data, param):
    while True:
        if cfg['local'] in received_data and received_data[cfg['local']] != None:
            print("Obtain data")
            break
        elif all([received_data[ip] for ip in cfg['prev_nodes']]):
            if len(cfg['prev_nodes']) > 1:
                received_data[cfg['local']] = tensor_comparision(cfg, received_data)
            else:
                received_data[cfg['local']] = received_data[cfg['prev_nodes'][0]]
            print(f"The best replica is from {param['choice']}")
            print("Processed")
            break

    inference_procedure(cfg, received_data, param)


# Simulate Distributed Training
def process_data(cfg, received_data, param):
    while True:
        if cfg['local'] in received_data:
            received_data = pickle.loads(received_data[cfg['local']])
            param['choice'] = cfg['local']
            print("Obtain data")
            break
        else:
            if all([received_data[ip] for ip in cfg['prev_nodes']]):
                # data_adversarial(cfg, received_data)
                if len(cfg['prev_nodes']) > 1:
                    param["choice"] = tensor_comparision(cfg, received_data)
                else:
                    param["choice"] = cfg['prev_nodes'][0]
                print(f"The best replica is from {param['choice']}")
                print("Processed")
                break


def data_adversarial(cfg, received_data):
    # Adding adversarial noise to the received data
    for ip in cfg['prev_nodes']:
        data = pickle.loads(received_data[ip]) # tensor
        data += 0.1 * torch.randn(100, 100)    # for testing
        received_data[ip] = pickle.dumps(data)
    print("Noise model output")


def tensor_comparision(cfg, received_data, threshold = 0.9):
    # Using one by one cosine similarity checking. (Time consuming)
    # Good for small models, but not large model.
    rep_data = [pickle.loads(received_data[ip]) for ip in cfg['prev_nodes']]

    # Permutation similarity
    similarity12 = F.cosine_similarity(rep_data[0], rep_data[1])
    similarity13 = F.cosine_similarity(rep_data[0], rep_data[2])
    similarity23 = F.cosine_similarity(rep_data[1], rep_data[2])

    assert similarity12.size() == similarity13.size() and similarity12.size() == similarity23.size()

    # avg_similarity = [ (similarity12 + similarity13) / 2,
    #                    (similarity12 + similarity23)/ 2 ,
    #                    (similarity13 + similarity23) / 2]
    #
    print(similarity12)   # 100 dim
    print(similarity13)
    print(similarity23)

    # diff_12 = torch.abs(avg_similarity[0] - avg_similarity[1]) < threshold
    # diff_13 = torch.abs(avg_similarity[0] - avg_similarity[2]) < threshold
    # diff_23 = torch.abs(avg_similarity[1] - avg_similarity[2]) < threshold
    diff_12 = similarity12 > threshold
    diff_13 = similarity13 > threshold
    diff_23 = similarity23 > threshold

    # if false exists, it means one of them is far away than others
    best_index = None

    if torch.all((diff_12 | diff_13 )):
        best_index = 0
    else:
        false_indices = torch.nonzero(~(diff_12 | diff_13 ))
        f"Tensor 1 has suspicious changed"

    if torch.all((diff_12 | diff_23 )):
        best_index = 1
    else:
        false_indices = torch.nonzero(~(diff_12 | diff_13 ))
        f"Tensor 2 has suspicious changed"

    if torch.all((diff_13 | diff_23 )):
        best_index = 2
    else:
        false_indices = torch.nonzero(~(diff_12 | diff_13 ))
        f"Tensor 3 has suspicious changed"

    if not best_index:
        best_index = 0
        print("All replicas could be changed!")

    return cfg['prev_nodes'][best_index]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Communication")
    parser.add_argument('--root', help='The root ip of this communication', default="169.234.9.247")
    parser.add_argument('--local', help='The local ip of this machine', default="127.0.0.1")
    parser.add_argument('--prevNodes', nargs='+',  help='The ip lists of previous nodes  ', default=[])
    parser.add_argument('--nextNodes', nargs='+',  help='The ip lists of next nodes  ', default=[])
    parser.add_argument('-t', '--com_type', help='Type of Message Queue Type', default=zmq.DEALER)
    parser.add_argument('-p', '--port', help='communciation port', default=12345)
    parser.add_argument('--rootPort', help='communciation root port', default=23456)


    args = parser.parse_args()

    cfg = {
        "local": args.local,
        "root": args.root,
        "root_port": args.rootPort,
        "prev_nodes": args.prevNodes,
        "next_nodes": args.nextNodes,
        "port": args.port
    }

    #cfg = yaml.load(open('./config.yaml', 'r'), Loader=yaml.Loader)

    start_time = time.time()

    context = zmq.Context()

    received_data = {}

    current_batch = 0

    socks = {}
    param = {
             "model_path": os.path.expanduser("~/SecureConnection/onnx_model/bloom560m/module.zip"),
             # "model_path": os.path.expanduser("~/SecureConnection/splitted_models/resnet/model.pkl"),
             "status": '',
             "choice": '',
             "num_batch": 0,
             "onnx": True,
             "model_name": "bloom-560m",
            }

    # Start connecting with root
    param['status'] = 'Ready'

    # event1 = threading.Event()
    root_sock = client.establish_connection(context, zmq.DEALER, cfg['root_port'], cfg['root'])
    # root_thread = threading.Thread(target=client.communication_open_close, args=(cfg, root_sock, param, context, socks))
    # root_thread.start()
    client.communication_open_close(cfg, root_sock, param)

    prepare_time = None
    tokenizer = None
    if param["model_name"] == "bloom-560m":
        model_name = "bigscience/bloom-560m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if param['status'] == "Running":
        # Set up all tcp connection of this machine for prev and next nodes
        for ip in cfg['prev_nodes']:
            print(f"Tcp connnection to prev nodes {ip} set up")
            socks[ip] = client.establish_connection(context, zmq.DEALER, cfg['port'], ip)
            received_data[ip] = None  # initial received msg

        next_sock = server.establish_connection(context, zmq.ROUTER, cfg['port'])
        for ip in cfg['next_nodes']:
            print(f"Tcp connnection to next nodes {ip} set up")
            socks[ip] = next_sock


        prepare_time = time.time()
        print(f"Prepare Time usage: {prepare_time - start_time}")
        # High level message transmission
        # Multi-threading for all message channels open
        while current_batch < param["num_batch"]:

            threads = []
            lock = threading.Lock()

            for ip in cfg['prev_nodes']:
                print(f"Prev nodes {ip} communication start:")
                t = threading.Thread(target=client.mt_communication_type5, args=(ip, socks[ip], received_data, lock))
                threads.append(t)

            # Inital data loading
            if len(cfg['prev_nodes']) == 0:
                # No Server, then initial the Tensor data
                data = [torch.zeros(8, 3, 244, 244), torch.ones(8, 3, 244, 244)]
                print("Data Created")

                if param["onnx"]:
                    if param["model_name"] == "bloom-560m":
                        data = [tokenizer("I love distributed learning on edge!", return_tensors="pt")[
                            "input_ids"]]

                    received_data[cfg['local']] = pickle.dumps([data[current_batch].numpy()])
                else:
                    received_data[cfg['local']] = pickle.dumps(data[current_batch])

            for ip in cfg['next_nodes']:
                print(f"Next nodes {ip} communication start:")
                t = threading.Thread(target=server.mt_communication_type5, args=(ip, socks[ip], received_data, param, lock))
                threads.append(t)

            t_process = threading.Thread(target=processing, args=(cfg, received_data, param))
            threads.append(t_process)

            for i in threads:
                i.start()

            for t in threads:
                t.join()

            current_batch += 1

            print(pickle.loads(received_data[cfg['local']]))
            if param["onnx"]:
                print(f"Output size: {len(pickle.loads(received_data[cfg['local']]))}")
            else:
                print(f"Output size: {pickle.loads(received_data[cfg['local']]).size()}")

            # Buffer clean up
            param['choice'] = ''
            received_data[cfg['local']] = None   # processed data
            for ip in cfg['prev_nodes']:         # received data
                received_data[ip] = None

        param['status'] = "Finish"
        print(f"Running Time usage: {time.time() - prepare_time}")

        # root_thread.join()
    # client.communication_open_close(socks[cfg['root']], param, context, socks) # Automatically close all channel

    if param['status'] == 'Finish':
        client.communication_open_close(cfg, root_sock, param, context, socks) # Automatically close all channel
        # root_sock.close()
        # context.term()


        # for ip in cfg['prev_nodes']:
        #     t = client.multi_threading_communication(1, client.communication_type5, send_socks, received_data)


    # if len(cfg["prev_nodes"]) > 0:
    #     client.multi_threading_communication(1, client.communication_type5, send_socks, received_data)
    #
    #     received_serial_data,  rece_socks = communication_with_prev_nodes(cfg, context)
    #     socks.extend(rece_socks.values())
    #
    #     for ip in cfg['prev_nodes']:
    #         received_data = pickle.loads(received_serial_data[ip][0])  # process received data
    #         print(received_data)
    # else:
    #     # No Server, then initial the Tensor data
    #     received_data = torch.rand(4,10)
    #
    # # Start process received data
    # if received_data != None:
    #     received_data = process_data(received_data)
    #
    # # Start process received data
    # if cfg['next_nodes'].size() > 0:
    #     send_socks = communication_with_next_nodes(cfg, context, received_data)
    #     socks.extend(send_socks)

    # close_sockets(socks)
    # context.term()
