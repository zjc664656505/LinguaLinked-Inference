## This File remote kill the communication process, if it fails

from control import remote_shell, kill_process

if __name__ == "__main__":
    root_ip = "128.195.41.60"

    # communication_layers = [
    #     ["128.195.41.49"],
    #     ["128.195.41.33", "128.195.41.61", "128.195.41.62"],
    #     ["128.195.41.35", "128.195.41.28"]
    # ]

    communication_layers = [
        ["128.195.41.53"],
        ["128.195.41.55"],
        ["128.195.41.57"]
    ]

    ips = communication_layers[0] + communication_layers[1] + communication_layers[2]
    results = {i: [] for i in ips}
    threads = []

    # shells_l0 = remote_shell(communication_layers[0], username='yurun', passwd='uitim1', port=22)
    shells_l0 = remote_shell(communication_layers[0])
    shells_l1 = remote_shell(communication_layers[1])
    shells_l2 = remote_shell(communication_layers[2])

    for sh in shells_l0 + shells_l1 + shells_l2:
        kill_process(sh)
