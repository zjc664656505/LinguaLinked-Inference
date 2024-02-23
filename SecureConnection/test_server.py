import socket

def start_server():
    server_ip = '192.168.1.6'
    server_port = 23456

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((server_ip, server_port))
        s.listen()
        print(f"Server listening on {server_ip}:{server_port}")
        
        conn, addr = s.accept()
        with conn:
            print(f"Connected by: {addr}")
            data = conn.recv(1024)
            print(f"Received data: {data.decode()}")

if __name__ == "__main__":
    start_server()
