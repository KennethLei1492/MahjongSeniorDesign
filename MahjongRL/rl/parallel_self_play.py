"""Parallel self-play: N worker processes feed one PPO learner.

Each worker owns a CPU copy of the network and plays complete games with
torch limited to a single thread (so 8 workers use 8 cores cleanly instead
of fighting over them). The learner broadcasts fresh weights every
iteration and gathers the workers' trajectory batches.

train.py uses this in a PIPELINED loop: while the learner runs the PPO
update on iteration k, the workers are already playing iteration k+1 with
the pre-update weights. The behavior policy's logprobs are stored with each
trajectory, so PPO's importance ratios stay correct despite the one-step
staleness - standard actor-learner practice.

Windows note: uses the 'spawn' start method (the only one on Windows), so
this module must stay importable without side effects, and any script using
ParallelCollector needs the `if __name__ == "__main__":` guard train.py has.
"""
import multiprocessing as mp

import numpy as np
import torch


def _worker_main(worker_id, task_q, result_q, model_kwargs, min_faan,
                 reward):
    torch.set_num_threads(1)  # one core per worker; the learner gets the rest
    from rl.model import build_model
    from rl.self_play import collect_batch

    model = build_model("cpu", **(model_kwargs or {}))
    model.eval()
    while True:
        task = task_q.get()
        if task is None:
            return
        weights, num_games, seed_base = task
        if weights is not None:
            model.load_state_dict(weights)
        batch, results = collect_batch(model, num_games, "cpu", seed_base,
                                       min_faan=min_faan, reward=reward)
        result_q.put((batch, results))


class ParallelCollector:
    def __init__(self, num_workers, model_kwargs=None, min_faan=0,
                 reward="win"):
        self.n = num_workers
        ctx = mp.get_context("spawn")
        self.task_qs = [ctx.Queue() for _ in range(num_workers)]
        self.result_q = ctx.Queue()
        self.procs = [
            ctx.Process(target=_worker_main,
                        args=(i, self.task_qs[i], self.result_q, model_kwargs,
                              min_faan, reward),
                        daemon=True)
            for i in range(num_workers)]
        for p in self.procs:
            p.start()
        self._outstanding = 0

    def dispatch(self, state_dict, num_games, seed_base):
        """Send weights + game quotas to all workers (non-blocking)."""
        cpu_sd = {k: v.detach().cpu() for k, v in state_dict.items()}
        per = max(1, num_games // self.n)
        assigned = 0
        for i, q in enumerate(self.task_qs):
            games = (num_games - assigned) if i == self.n - 1 else \
                min(per, num_games - assigned)
            if games <= 0:
                continue
            # distinct seed blocks so workers never replay each other's games
            q.put((cpu_sd, games, seed_base + i * 1_000_000))
            assigned += games
            self._outstanding += 1

    def gather(self):
        """Block until all dispatched workers report; return merged batch."""
        batches, results = [], []
        for _ in range(self._outstanding):
            b, r = self.result_q.get()
            batches.append(b)
            results.extend(r)
        self._outstanding = 0
        merged = {k: np.concatenate([b[k] for b in batches])
                  for k in batches[0]}
        return merged, results

    def stop(self):
        for q in self.task_qs:
            q.put(None)
        for p in self.procs:
            p.join(timeout=10)
