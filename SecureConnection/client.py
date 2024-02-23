import threading
import zmq
import time
import os
import zipfile


from encryption import generate_keys, serialize_public_key, deserialize_public_key, decrypt_session_key, \
    serialize_private_key, deserialize_private_key, decrypt_large_zip, sess_encrypt, public_encrypt, decrypt_file

client_key = {"public_key":None, "private_key":None}
server_key = {"public_key":None, "session_key": None}

def initial():
    if  not client_key['public_key']:
        client_private_key, client_public_key = generate_keys()
        client_key['public_key'] =  serialize_public_key(client_public_key)
        client_key['private_key'] = serialize_private_key(client_private_key)

def session_establish(sock):
    sock.send(client_key["public_key"])
    server_public_pky, encrypted_session_key = sock.recv_multipart()
    session_key = decrypt_session_key(deserialize_private_key(client_key["private_key"]), encrypted_session_key)
    server_key["public_key"] = server_public_pky
    server_key["session_key"] = session_key

    print(client_key)
    print(server_key)


def receive_file(path, sock, size=2048):
    with open(path, 'wb') as f:
        data = sock.recv(size)
        while data:
            f.write(data)
            print("Data is writen")
            data = sock.recv(size)


def establish_connection(context, type, port, address='localhost'):
    socket = context.socket(type)
    socket.connect(f"tcp://{address}:{port}")
    return socket

def communication_type1_reply(client_id, sender, receiver):
    binary_data = sender.recv()
    print(f"Client {client_id} received binary data: {binary_data}")
    receiver.send_string(f"Client {client_id} has received the data")

# Zip file communication
def communication_type2_reply(client_id, output_path, sender):
    # Send binary data to clients
    # sender.setsockopt_string(zmq.SUBSCRIBE, "DATA")
    sender.send_string("SUBSCRIBE")
    with open(output_path, 'wb') as f:
        data = sender.recv(4096)
        while data:
            if data == b'end':
                break
            f.write(data)
            print("Data is writen")
            data = sender.recv(4096)
    print("Received")


# Encrypted single Zip file communication
def single_communication_type4_reply(id, encrypt_file_path, sender):
    print(public_encrypt(server_key['public_key'], bytes(id, 'utf8')))
    sender.send_multipart([public_encrypt(server_key['public_key'], bytes(id, 'utf8'))])
    session_key = get_session_key()
    # Todo get_group_session_key()
    response = sender.recv_multipart()
    file_data = response[0]
    with open(encrypt_file_path +".encrypt", "wb") as f:
        f.write(file_data)
    decrypt_file(encrypt_file_path +".encrypt", encrypt_file_path, session_key)


def get_session_key():
    return server_key['session_key']

# Encrypted Large Zip file communication
def single_communication_type3_reply(id, encrypt_file_path, sender):
    session_key = get_session_key()
    print(public_encrypt(server_key['public_key'], bytes(id, 'utf8')))
    sender.send_multipart([public_encrypt(server_key['public_key'], bytes(id, 'utf8'))])
    # Todo get_group_session_key()
    communication_type2_reply(1, encrypt_file_path + ".encrypt", sender)
    decrypt_large_zip(encrypt_file_path + ".encrypt", encrypt_file_path, session_key)


# Tensor Data communication
def communication_type5(receiver, data):
    # data = pickle.dumps()
    # data = pickle.loads()
    ip, rece = receiver
    rece.send(b"Data")
    msg = rece.recv()
    data[ip].append(msg)
    print(f"Received data {msg}")

# Tensor Data communication
def mt_communication_type5(target_ip, receiver, received_data, lock):
    # data = pickle.dumps()
    # data = pickle.loads()
    with lock:
        # while True:
        receiver.send(b"Request data")
        msg = receiver.recv()
        received_data[target_ip] = msg
        print(f"Received data")


def communication_open_close(cfg, receiver, param, context=None, all_sockets=None):
    ## Status: Ready, Open, Prepare, Initialized, Start, Running, Finish
    ## Key status: Ready, Running, Finish
    while True:
        ## Ready
        if param["status"] == "Ready":
            print("Status: Ready")
            receiver.send_multipart([b"Ready", cfg["local"].encode('utf-8')])
            msg = receiver.recv()
            ## Open
            if msg == b"open":
                param['status'] = "Open"
                print("status: Open")
            ## Prepare
            msg = receiver.recv()
            if msg == b"Prepare":
                communication_prepare(receiver, param)

            ## Initialized
            model_initialize(param)
            param['status'] = "Initialized"
            print("status: Initialized")
            receiver.send(b"Initialized")

            ## Start
            msg = receiver.recv()
            print(msg)
            if msg == b"Start":
                param['status'] = "Start"
                print("status: Start")
                param['num_batch'] = int(receiver.recv().decode('utf-8'))
                print(param["num_batch"])
                ## Header receive the msg from root
                ## Tail sends the msg to the root

                ## Running
                receiver.send(b"Running")
                print("status: Running")
                param['status'] = "Running"
                break

        elif param["status"] == "Finish":
            print("Finish task")
            receiver.send(b'Finish')
            msg = receiver.recv()
            print(msg)
            print("status: Close")
            if  msg == b'Close' and all_sockets != None:
                close_sockets(all_sockets)
            break

        elif param["status"] == "Running":
            # print("status: Running")
            # time.sleep(2)
            pass

def receive_model_file(path, sock, chunked=True, chunk_size=1024*1024):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    if not chunked:
        with open(path, 'wb') as f:
            data = sock.recv()
            f.write(data)
            print("Data is writen")
    else:
        received_data = b""
        while True:
            chunk = sock.recv()
            received_data += chunk
            # if len(chunk) < chunk_size:
            if chunk == b"":
                break
        with open(path, "wb") as file:
            file.write(received_data)
        print("Data is writen")


from test_inference_resnet import initialize_model
from test_inference_resnet_onnx import initialize_onnx_model

def model_initialize(param):
    if param["onnx"]:
        param["model"], param["model_input_name"] = initialize_onnx_model(os.path.dirname(param["model_path"]))
    else:
        param["model"] = initialize_model(param["model_path"])

def communication_prepare(sender, param):
    param['status'] = "Prepare"
    print("status: Prepare")
    msg = sender.recv()
    if not (msg == b'True'): # Skip the model, causing model exists
        receive_model_file(param["model_path"], sender)
        print("Model Received")
    else:
        print("Model Exists")
    if param["onnx"]:
        unzip_file(param["model_path"])

def unzip_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"The zip file '{file_path}' does not exist.")
        return
    # Check if the file is a zip file

    while not zipfile.is_zipfile(file_path):
        print(f"'{file_path}' is not a zip file. (Packet loss)")
        time.sleep(5)

    # Unzip the file
    target_directory = os.path.dirname(file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(target_directory)
        print(f"The file '{file_path}' has been successfully unzipped to '{target_directory}'.")



def multi_threading_communication(num_clients, communication_type, receivers, data, receive_from_mt = False):

    threads = []

    for rece in receivers.items():
        if not receive_from_mt:
            # Receive from the different ip one times
            t = threading.Thread(target=communication_type, args=(rece, data))
            threads.append(t)
        else:
            # Receive from the same ip mult-thread data msg, need lock
            lock = threading.Lock()
            for i in range(num_clients):
                t = threading.Thread(target=communication_type, args=(rece, data, lock))
                threads.append(t)

    for i in threads:
        i.start()

    for t in threads:
        t.join()


def close_sockets(sockets):
    for s in sockets.values():
        s.close()


def close_sockets_(sockets):
    for s in sockets:
        s.close()




# if __name__ == "__main__":
#     start = time.time()
#     context = zmq.Context()
#     client_id = "1"
    # initial()

    '''Test 1 session_establish'''
    # sock = establish_connection(context, zmq.REQ, 12345)
    # session_establish(sock)
    # sock.close()

    '''Test 2 send uni-directional binary data via '''
    # send = establish_connection(context, zmq.PULL, 5557)
    # rece = establish_connection(context, zmq.PUSH, 5558)
    # communication_type1_reply(client_id, send, rece)
    # send.close()
    # rece.close()
    #
    '''Test 3 send large zip data via Router & Dealer'''
    # rece = establish_connection(context, zmq.DEALER, 5559)
    # communication_type2_reply(client_id, './Test/Client1/received_data.zip', rece)
    # rece.close()
    #
    # # os.remove('./Test/Client1/received_data.zip')
    #
    '''Test 4 send encrypted large zip file'''
    # send = establish_connection(context, zmq.DEALER, 5556)
    # single_communication_type3_reply(client_id, "./Test/client1/data.zip", send)
    # send.close()
    #
    # # os.remove('./Test/Client1/data.zip')
    # # os.remove('./Test/Client1/data.zip.encrypt')


    '''Test 5 send encrypted whole zip file'''
    # send = establish_connection(context, zmq.DEALER, 5557)
    # single_communication_type4_reply(client_id, "./Test/client1/data.zip", send)
    # send.close()

    '''Test 6 send Tensor Data'''

    # ips = ['169.234.61.40', '169.234.44.179', '169.234.4.155']
    # ips = ['127.0.0.1']
    # received_data = {}
    # send_socks = {}
    #
    # for ip in ips:
    #     send_socks[ip] = establish_connection(context, zmq.DEALER, 12345, ip)
    #     received_data[ip] = []
    #
    # multi_threading_communication(1, communication_type5, send_socks, received_data)
    #
    # import pickle
    #
    # for ip in ips:
    #     data = pickle.loads(received_data[ip][0])
    #     print(data)
    #
    # close_sockets(send_socks)
    # context.term()
    #
    # print(f"Time usage: {time.time() - start}")
