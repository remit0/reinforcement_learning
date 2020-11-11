import gym
import threading
from torch import multiprocessing as mp

from a3c.nets import create_networks
from a3c.worker import Worker


if __name__ == '__main__':

    policy_net, value_net = create_networks()
    policy_net.share_memory()
    value_net.share_memory()

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # mp.cpu_count()
    for worker_id in range(1):
        worker = Worker(name=f"worker_{worker_id}", env=gym.make("CartPole-v0").env, gamma=0.99)
        p = threading.Thread(target=worker.run, args=(policy_net, value_net, 1e-4, 1e-3, 10, counter, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
