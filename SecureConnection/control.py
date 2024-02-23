import spur
import threading
import re

'''
This file run on the root machine, to start the decentralized communication.
'''

def remote_shell(ips, username='userland', passwd: str="kimvjrm", port= 2022):
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
    # print(shell_result.output)
    results[ip].append(shell_result.output.decode('utf-8').strip('\n'))


def kill_process(remote_shell):
    cmd = "ps aux|grep communication.py"
    print(cmd.split(" "))
    shell_result = remote_shell.run(['sh','-c', "ps aux|grep communication.py"])
    pids = shell_result.output.decode('utf-8').split('\n')
    match = re.search(r'\d+', pids[0])
    kill_cmd = f"kill {int(match.group())}"
    try:
        remote_shell.run(kill_cmd.split(" "))
    except:
        print("Fail")


if __name__ == "__main__":
    root_ip = "128.195.41.60"

    communication_layers = [
        ["128.195.41.53"],
        ["128.195.41.55"],
        ["128.195.41.57"]
    ]

    ips = communication_layers[0] + communication_layers[1] + communication_layers[2]
    results = {i:[] for i in ips}
    threads = []

    # shells_l0 = remote_shell(communication_layers[0], username='yurun', passwd='uitim1', port=22)
    shells_l0 = remote_shell(communication_layers[0])
    shells_l1 = remote_shell(communication_layers[1])
    shells_l2 = remote_shell(communication_layers[2])

    # Layer 0
    for ip, shell in zip(communication_layers[0], shells_l0):
        next_nodes = " ".join(communication_layers[1])
        # cmd = f"python3 /home/yurun/SecureConnection/communication.py --local {ip} --root {root_ip} --nextNodes {next_nodes}"
        cmd = f"python3 /home/userland/SecureConnection/communication.py --local {ip} --root {root_ip} --nextNodes {next_nodes}"
        print(cmd)
        t_host = threading.Thread(target=secure_connection, args=[shell, ip, results, cmd])
        threads.append(t_host)

    # Layer 1
    for ip, shell in zip(communication_layers[1], shells_l1):
        next_nodes = " ".join(communication_layers[2])
        prev_nodes = " ".join(communication_layers[0])
        cmd = f"python3 /home/userland/SecureConnection/communication.py --local {ip} --root {root_ip} --prevNodes {prev_nodes} --nextNodes {next_nodes}"
        print(cmd)
        t1 = threading.Thread(target=secure_connection, args=[shell, ip, results, cmd])
        threads.append(t1)

    # Layer 2
    for ip, shell in zip(communication_layers[2], shells_l2):
        prev_nodes = " ".join(communication_layers[1])
        cmd = f"python3 /home/userland/SecureConnection/communication.py --local {ip} --root {root_ip} --prevNodes {prev_nodes}"
        print(cmd)
        t = threading.Thread(target=secure_connection, args=[shell, ip, results, cmd])
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