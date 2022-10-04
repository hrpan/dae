import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env import DummyVecEnv


def _worker(
    remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper
) -> None:
    # Import here to avoid a circular import

    from algo.util import register_envs
    register_envs()

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(env.env_is_wrapped(data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class CustomVecEnv(VecEnv):
    """
    Asynchronous vectorized environment with which distributes the number of environments
    according to the specified threads. This is useful when the number of parallel
    environments is much larger than the number of physical processors.

    WARNING: ONLY CORE FUNCTIONALITY IS IMPLEMENTED

    Code adapted from:
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    :param threads: number of parallel threads
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None, threads: int = 1):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.threads = min(threads, len(env_fns))
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.threads)])
        self.processes = []
        self.slicing = []
        q, r = divmod(len(env_fns), self.threads)
        l = 0
        for thread in range(self.threads):
            n = q + (1 if thread < r else 0)
            self.slicing.append(slice(l, l + n))
            l += n
        for work_remote, remote, sli in zip(self.work_remotes, self.remotes, self.slicing):
            def wrapper():
                return DummyVecEnv(env_fns[sli])
            args = (work_remote, remote, CloudpickleWrapper(wrapper))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, sli in zip(self.remotes, self.slicing):
            remote.send(("step", actions[sli]))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.concatenate(obs), np.concatenate(rews), np.concatenate(dones), tuple(sum(infos, []))

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        for idx, (remote, sli) in enumerate(zip(self.remotes, self.slicing)):
            remote.send(("seed", seed + sli.start))
        return sum([remote.recv() for remote in self.remotes], [])

    def reset(self) -> VecEnvObs:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.concatenate(obs)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return sum(imgs, [])

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        raise NotImplementedError()

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        raise NotImplementedError()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        for remote in self.remotes:
            remote.send(("is_wrapped", wrapper_class))
        return sum([remote.recv() for remote in self.remotes], [])

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        raise NotImplementedError()
