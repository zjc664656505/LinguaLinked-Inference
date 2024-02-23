import time
import zmq
import threading

from .encryption import generate_keys, encrypt_session_key, serialize_public_key, deserialize_public_key, \
    serialize_private_key, encrypt_large_zip, private_decrypt, encrypt_file

id = 1
client_keys = {}
server_key = {"public_key":None, "private_key":None}


def initial():
    if  not server_key['public_key']:
        server_private_key, server_public_key = generate_keys()
        server_key['public_key'] =  serialize_public_key(server_public_key)
        server_key['private_key'] = serialize_private_key(server_private_key)

def session_establish(sock):
    global id
    client_pky = deserialize_public_key(sock.recv())
    session_key, encrypted_session_key = encrypt_session_key(client_pky)
    client_keys[str(id)] = {"key": serialize_public_key(client_pky), "session_key":session_key}
    sock.send_multipart([server_key["public_key"], bytes(encrypted_session_key)])
    id+=1
    print(client_keys)
    print(server_key)
    # sock.close()


def create_group_session_key(clients):
    pass


def send_file(path, sock, size=4096):
    with open(path, 'rb') as f:
        data = f.read(size)
        while data:
            sock.send(data)
            print("data is sent")
            data = f.read(size)

def establish_connection(context, type, port, address=None):
    socket = context.socket(type)
    socket.bind(f"tcp://*:{port}")
    # socket.setsockopt(zmq.LINGER, 1000)
    return socket


# Binary Data communication
def communication_type1(clients, sender, receiver):
    # Send binary data to clients
    data = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A"
    for i in range(clients):   ## send the binary
        sender.send(data)
        time.sleep(1)

    for _ in range(clients):   ## receive the string
        response = receiver.recv_string()
        print(f"Received response: {response}")


# Zip file communication
def communication_type2(num_clients, zip_file_path, sender):

    clients_expected = num_clients
    clients_received = 0

    while clients_received < clients_expected:
        # Check if there's a subscription request
        try:
            client_id, msg = sender.recv_multipart()
            print(client_id)
            msg = msg.decode('utf-8')
            if msg.startswith("SUBSCRIBE"):
                clients_received += 1
                # send_file("./Test/Server/data.zip", sender)
                with open(zip_file_path, 'rb') as f:
                    data = f.read(4096)
                    while data:
                        sender.send_multipart([client_id, data])
                        print("data is sent")
                        data = f.read(4096)
                    sender.send_multipart([client_id, b'end'])
                print("Data sent to client", clients_received)

        except zmq.Again:
            pass
        # Wait a bit before checking again
        time.sleep(0.1)


def get_session_key(id):
    return client_keys.get(id)['session_key']

def verification(sender):
    # Todo ZMQ is not able to handle communication verification,
    #  need addition library to handle. Here we just assume verification success
    client_id, msg = sender.recv_multipart()
    id = private_decrypt(server_key['private_key'], msg)
    return client_id, id.decode('utf-8')


# Encrypted single Zip file communication
def single_communication_type4(zip_file_path, sender):
    client_id, id  = verification(sender)
    session_key  = get_session_key(id)
    print(session_key)
    encrypt_file(zip_file_path, zip_file_path + ".encrypt", session_key)
    with open(zip_file_path +".encrypt", "rb") as f:
        zip_data = f.read()
    message = [client_id, zip_data]
    sender.send_multipart(message)


# Encrypted Large Zip file communication
# Todo: multi-threads to communicate encrpyted data
def single_communication_type3(zip_file_path, sender):
    client_id, id  = verification(sender)
    session_key  = get_session_key(id)
    print(session_key)
    # Todo get_group_session_key()
    encrypt_large_zip(zip_file_path, zip_file_path + ".encrypt", session_key)
    communication_type2(1, zip_file_path + ".encrypt", sender)


# Tensor Data communication
def communication_type5(sender, data):
    # data = pickle.dumps()
    # data = pickle.loads()
    try:
        client_id, msg = sender.recv_multipart()
        msg = msg.decode('utf-8')
        if msg.startswith("Data"):
            # send_file("./Test/Server/data.zip", sender)
            sender.send_multipart([client_id, data])
            print(f"Data sent to client {client_id}")
    except zmq.Again:
        pass

def mt_communication_type5(target_ip, sender, received_data, param, lock):
    # data = pickle.dumps()
    # data = pickle.loads()
    try:
        client_id, msg = sender.recv_multipart()
        msg = msg.decode('utf-8')
        if msg.startswith("Request data"):
            print(param['choice'])
            while True:
                if param['choice'] != '' and len(received_data[param['choice']]) > 0 :
                    with lock:
                        sender.send_multipart([client_id, received_data[param['choice']]])
                    print(f"Data sent to client {target_ip}")
                    break
                else:
                    # print("Waiting ...")
                    time.sleep(0.01)
    except zmq.Again:
        pass


# multi threading for unrestricted communication type and single port
def multi_threading_communication(num_clients, mt_communication_type, data, sock):
    lock = threading.Lock()

    threads = []
    for i in range(num_clients):
        t = threading.Thread(target=mt_communication_type, args=(sock, data, lock))
        threads.append(t)

    for i in threads:
        i.start()

    for t in threads:
        # if threading.active_count() == 1:
            # all threads are done, set the Event object
        t.join()

def communication_open_close(sender, config, status, conditions, lock=True, open=True):
    ## Status: Ready, Open, Prepare, Initialized, Start, Running, Finish
    while True:
        with lock:
            info = sender.recv_multipart()
        client_id = info[0]
        msg = info[1]
        print(client_id)
        print(msg)
        ## Ready
        if open and msg == b'Ready':
            ## Open
            if len(info) != 3:
                print("Error")

            config["ids"][client_id] = info[2]
            print(config["ids"])

            sender.send_multipart([client_id, b'Open'])
            status[client_id] = b'Open'
            print(f"Status: Open {config['ids'][client_id]}")

            ## Prepare
            communication_prepare(sender, config, client_id, status)
            print(f"Status: Prepare {config['ids'][client_id]}")

        ## Initialized
        elif msg == b'Initialized':
            status[client_id] = b'Initialized'
            print(f"Status: Initialized {config['ids'][client_id]}")

            # while True:
            #     if check_status(status, num_devices, b"Initialized"):
            #         break
            #     time.sleep(0.1)

            with conditions[0]:
                while not check_status(status, config, b"Initialized"):
                    conditions[0].wait()
                conditions[0].notify_all()

            ## Start
                sender.send_multipart([client_id, b"Start"])
                status[client_id] = b'Start'
                print(f"Status: Start {config['ids'][client_id]}")
            # print(status)

        elif msg == b"Running":
            # Todo Send data to the header、 machine, wait result from tailer machine
            pass

        elif msg == b'Finish':
            status[client_id] = b'Close'

            # with lock:
            #     while not check_status(status, num_devices, b"Close"):
            #         condition.wait()
            #
            #     if check_status(status, num_devices, b"Close"):
            #         condition.notifyAll()

            # while True:
            #     if check_status(status, num_devices, b"Close"):
            #         break
            #     time.sleep(1)

            with conditions[1]:
                while not check_status(status, config, b"Close"):
                    conditions[1].wait()
                conditions[1].notify_all()

                sender.send_multipart([client_id, b"Close"])
            print(f"Close {config['ids'][client_id]}")
            break

    # while True:
    #     if len(status) == num_devices and check_status(status, b"Close"):
    #         sender.send_multipart([client_id, b"Close"])
    #         print(f"Close {client_id}")
    #         break
    #     time.sleep(1)


def send_model_file(path, sock, client_id, size=4096):
    with open(path, 'rb') as f:
        data = f.read()
        sock.send_multipart([client_id, data])
        print("model data is sent")

def communication_prepare(sender, config, client_id, status):
    sender.send_multipart([client_id, b'Prepare'])
    node_ip = config["ids"][client_id]
    print(f"send {config['file_path'][node_ip]} to {node_ip}")
    send_model_file(config["file_path"][node_ip], sender, client_id)
    status[client_id] = b"Prepare"


def communication_data_transmission(sender, num_devices, head_client_id, status):
    while check_status(status, num_devices, b"Start"):
        pass


def communication_result_transmission(sender, result, num_devices, tail_client_id, status):
    # while check_status(status, num_devices, b"Finish"):
    sender.send_multipart([b"res", result])
    pass




# def communication_pre_processing(sender, num_con, status, lock=True):
#     num_processing = 0
#     while True:
#         for client_id, stat in status:
#             if stat == b"Open":
#                 send_model_file("model.txt", sender, client_id)
#                 status['client_id'] = b"Processing"
#                 num_processing += 1
#         if num_processing == len(status):
#             break





    # with lock:
    #     while True:
    #         client_id, msg = sender.recv_multipart()
    #         print(client_id)
    #         if open and msg == b'Ready':
    #             status[client_id] = b"Open"
    #             break
    # while True:
    #     if len(status) == num_con and check_close(status):
    #         sender.send_multipart([client_id, b"Open"])
    #         print(f"Open {client_id}")
    #         break
    #
    # with lock:
    #     while True:
    #         client_id, msg = sender.recv_multipart()
    #         if msg == b'Finish':
    #             status[client_id] = b"Close"
    #             break
    #
    # while True:
    #     if len(status) == num_con and check_close(status):
    #         sender.send_multipart([client_id, b"Close"])
    #         print(f"Close {client_id}")
    #         break
    #     # time.sleep(1)

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


# Encrypted Signed Zip file communication
# def main():
#     # Todo multi-threads send signed msg to all
#     pass


# if __name__ == "__main__":
#     start = time.time()
#     context = zmq.Context()
#     initial()

    # threads_done = threading.Event()

    # '''Test 1 session_establish'''

    # '''Should we use REP to make key exchange ?'''
    # sock1 = establish_connection(context, zmq.REP, 12345)
    # multi_threading_communication(2, session_establish, sock1)
    # sock1.close()

        # if threading.active_count() == 2:
        #     # all threads are done, set the Event object
        #     threads_done.set()

    # time.sleep(0.1)

    # '''Test 2 send uni-directional binary data via '''
    ## Both user connect to one socket, for receive PUSH msg
    # send = establish_connection(context, zmq.PUSH, 5557)
    # rece = establish_connection(context, zmq.PULL, 5558)
    # communication_type1(2, send, rece)
    # rece.close()
    # send.close()

    # '''Test 3 send large zip data via Router & Dealer'''
    # send = establish_connection(context, zmq.ROUTER, 5559)
    # communication_type2(2, "./Test/Server/data.zip", send)
    # send.close()
    #
    # '''Test 4 send encrypted large zip file'''
    # send = establish_connection(context, zmq.ROUTER, 5556)
    # single_communication_type3("./Test/Server/data.zip", send)
    # send.close()
    #
    # send = establish_connection(context, zmq.ROUTER, 5557)
    # single_communication_type3("./Test/Server/data.zip", send)
    # send.close()

    # '''Test 5 send encrypted whole zip file'''
    # send = establish_connection(context, zmq.ROUTER, 5557)
    # single_communication_type4("./Test/Server/python.zip", send)
    # send.close()
    #
    # send = establish_connection(context, zmq.ROUTER, 5558)
    # single_communication_type4("./Test/Server/python.zip", send)
    # send.close()


    # '''Test 6 send Tensor Data'''
    # Router-Dealer is bidirectional, unrestricted Send/receive pattern





    # import torch
    # import pickle
    # data = pickle.dumps(torch.rand(4, 10))
    # # data = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A"
    #
    # send = establish_connection(context, zmq.ROUTER, 12345)
    # multi_threading_communication(1, mt_communication_type5, data, send)
    # print(data)
    #
    # send.close()
    # context.term()
    # print(f"Time usage: {time.time() - start}")




# Todo
# Multi-threads
# User verification
    # assign the id to the user at the time when the session is build, with session_key
    # verification needs user to send id public encrypted id for every socket build
# Communication via Signed file