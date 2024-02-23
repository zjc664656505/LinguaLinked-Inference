import spur
import threading

'''
This file run on the root machine, to start the decentralized communication.
'''

def remote_shell(ips, username='', passwd: str="kimvjrm", port= 8022):
    """
    Create remote shell for creating tcp connection object that can be referred for multiple times between server and client
    return: a list of spur objects -> list: [ssh_shell1, ssh_shell2, ...]
    """
    tcp_object = []
    for ip in ips:
        remote_shell = spur.SshShell(hostname=ip, username=username, port=port, password=passwd, missing_host_key=spur.ssh.MissingHostKey.warn)
        tcp_object.append(remote_shell)
    return tcp_object


def close_remote_shell(remote_shells):
    for shell in remote_shells:
        shell.close()


def secure_connection(remote_shell, ip, results, command):
    shell_result = remote_shell.run(command.split(" "))
    results[ip].append(shell_result.output.decode('utf-8').strip('\n'))

if __name__ == "__main__":

    # communication_layers = [
    #     ["192.168.1.12"],
    #     ["192.168.1.127", "192.168.1.128", "192.168.1.129"],
    #     ["192.168.1.130"]
    # ]

    communication_layers = [
        ["128.195.41.49"],
        ["169.234.39.235", "169.234.44.179", "169.234.4.155"],
        ["169.234.61.40"]
    ]

    ips = communication_layers[0] + communication_layers[1] + communication_layers[1] + communication_layers[2]
    results = {i:[] for i in ips}

    threads = []

    shells_l0 = remote_shell(communication_layers[0], username='yurun', passwd='uitim1', port=22)
    shells_l1 = remote_shell(communication_layers[1])
    shells_l2 = remote_shell(communication_layers[2])

    # Layer 0
    for ip, shell in zip(communication_layers[0], shells_l0):
        t_host = threading.Thread(target=secure_connection, args=[shell, ip, results, "python3 /home/yurun/SimpleSecureConnection/Server.py"])
        threads.append(t_host)

    # Layer 1
    for ip, shell in zip(communication_layers[1], shells_l1):
        t1 = threading.Thread(target=secure_connection, args=[shell, ip, results, 'python3 /data/data/com.termux/files/home/SimpleSecureConnection/Client1.py'])
        t2 = threading.Thread(target=secure_connection, args=(shell, ip, results, 'python3 /data/data/com.termux/files/home/SimpleSecureConnection/Server.py'))
        threads.append(t1)
        threads.append(t2)

    # Layer 2
    for ip, shell in zip(communication_layers[2], shells_l2):
        t = threading.Thread(target=secure_connection, args=(shell, ip, results, 'python3 /data/data/com.termux/files/home/SimpleSecureConnection/Client1.py'))
        threads.append(t)

    for i in threads:
        i.start()

    for t in threads:
        t.join()


    for k, v in results.items():
        print(f'machine {k}')
        print(f'Output log: ')
        for i in v:
            print(i)
        print("=========="*10)

    close_remote_shell(shells_l0)
    close_remote_shell(shells_l1)
    close_remote_shell(shells_l2)