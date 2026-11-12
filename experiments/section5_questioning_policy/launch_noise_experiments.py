"""
Launcher for the noise-robustness experiment (Appendix J).

Parallelism design:
  - ``--gpus 0 1 2 3`` and ``--mcts_workers_per_gpu N`` -> MCTS runs with
    ``len(gpus) * N`` concurrent workers, each pinned to one GPU via
    CUDA_VISIBLE_DEVICES set in the subprocess env.
  - ``--heuristic_workers K`` CPU workers for the heuristic methods.
  - MCTS and heuristic dispatchers run in parallel threads.
  - All jobs are idempotent: each runner skips if its output file exists.

Example 1 (full noise-robustness sweep across 4 GPUs):

    python launch_noise_experiments.py \\
        --gpus 0 1 2 3 \\
        --mcts_workers_per_gpu 4 \\
        --heuristic_workers 16

Example 2 (only a subset of heuristics):

    python launch_noise_experiments.py --skip_mcts \\
        --heuristic_methods H_RAI_d2 H_DVF H_RAND \\
        --heuristic_workers 16

Pass ``--dry_run`` to print the task plan without launching anything.
"""

import argparse
import os
import subprocess
import sys
import time
from queue import Queue
from threading import Thread, Lock

# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----



HEURISTIC_METHOD_SPECS = {
    "H_PWI_d1": ("H_O_X_d.py", {"measure": "PWI", "depth": 1}),
    "H_PWI_d2": ("H_O_X_d.py", {"measure": "PWI", "depth": 2}),
    "H_RAI_d1": ("H_O_X_d.py", {"measure": "RAI", "depth": 1}),
    "H_RAI_d2": ("H_O_X_d.py", {"measure": "RAI", "depth": 2}),
    "H_DVF":    ("H_DVF.py", {}),
    "H_RAND":   ("H_RAND.py", {}),
}


class ProgressCounter:
    """Thread-safe counter for printing [n/total] progress."""

    def __init__(self, total, tag):
        self.total = total
        self.tag = tag
        self.n = 0
        self.lock = Lock()
        self.t0 = time.time()

    def tick(self):
        with self.lock:
            self.n += 1
            n = self.n
        elapsed = time.time() - self.t0
        eta = elapsed / n * (self.total - n) if n > 0 else float("nan")
        print(
            f"[{self.tag} {n}/{self.total}] "
            f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min",
            flush=True,
        )


def expand_random_indices(spec):
    out = []
    for item in spec:
        if ".." in item:
            lo, hi = item.split("..")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(item))
    return sorted(set(out))


def build_mcts_tasks(args):
    tasks = []
    for method in ["ent", "var"]:
        for F3 in args.F3s:
            for T in args.Ts:
                for idx in args.random_indices:
                    tasks.append({
                        "script": args.mcts_script,
                        "method": method,
                        "T": T,
                        "random_index": idx,
                        "F3": F3,
                    })
    return tasks


def build_heuristic_tasks(args):
    tasks = []
    for method_name in args.heuristic_methods:
        if method_name not in HEURISTIC_METHOD_SPECS:
            raise ValueError(
                f"Unknown heuristic method: {method_name}. "
                f"Available: {list(HEURISTIC_METHOD_SPECS)}"
            )
        script, extra = HEURISTIC_METHOD_SPECS[method_name]
        for F3 in args.F3s:
            for idx in args.random_indices:
                tasks.append({
                    "script": script,
                    "T": args.heuristic_T,
                    "random_index": idx,
                    "F3": F3,
                    **extra,
                })
    return tasks


def task_to_cmd(task, script_dir, python_bin, gpu=None, results_root=None,
                save_timing=False, mcts_extra=None):
    cmd = [python_bin, os.path.join(script_dir, task["script"])]
    for key, val in task.items():
        if key == "script":
            continue
        cmd.extend([f"--{key}", str(val)])
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    if results_root:
        cmd.extend(["--results_root", results_root])
    if save_timing:
        cmd.append("--save_timing")
    if mcts_extra and task["script"] == "MCTS.py":
        cmd.extend(mcts_extra)
    return cmd


def worker_run_queue(queue, gpu, script_dir, python_bin, results_root,
                     log_prefix, counter, save_timing=False, mcts_extra=None):
    while True:
        task = queue.get()
        if task is None:
            queue.task_done()
            return
        cmd = task_to_cmd(task, script_dir, python_bin,
                          gpu=gpu, results_root=results_root,
                          save_timing=save_timing,
                          mcts_extra=mcts_extra)
        tag = f"[{log_prefix}]"
        print(f"{tag} $ {' '.join(cmd)}", flush=True)
        # Pin the GPU via subprocess env BEFORE Python/torch loads in the
        # child; setting os.environ inside the child after `import torch`
        # is unreliable because the CUDA runtime caches the visible device
        # list on first access.
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        try:
            subprocess.run(cmd, env=env, check=False)
        except Exception as ex:
            print(f"{tag} failed: {ex}", flush=True)
        counter.tick()
        queue.task_done()


def dispatch(tasks, worker_slots, script_dir, python_bin,
             results_root, log_prefix, save_timing=False, mcts_extra=None):
    """Launch len(worker_slots) workers; each slot is either an int (GPU id)
    or None (CPU)."""
    counter = ProgressCounter(len(tasks), log_prefix)
    q = Queue()
    for t in tasks:
        q.put(t)
    threads = []
    for i, gpu in enumerate(worker_slots):
        t = Thread(
            target=worker_run_queue,
            args=(q, gpu, script_dir, python_bin, results_root,
                  f"{log_prefix}-w{i}", counter),
            kwargs={"save_timing": save_timing, "mcts_extra": mcts_extra},
            daemon=True,
        )
        t.start()
        threads.append(t)
    for _ in worker_slots:
        q.put(None)
    for t in threads:
        t.join()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--mcts_workers_per_gpu", type=int, default=4,
                   help="Number of concurrent MCTS workers per GPU")
    p.add_argument("--heuristic_workers", type=int, default=16,
                   help="Number of parallel heuristic workers (CPU)")
    p.add_argument("--heuristic_T", type=int, default=8,
                   help="T value for heuristic runs (saved sequence length)")
    p.add_argument("--heuristic_methods", nargs="+",
                   default=list(HEURISTIC_METHOD_SPECS.keys()),
                   help=f"Subset of {list(HEURISTIC_METHOD_SPECS.keys())}")
    p.add_argument("--mcts_script", type=str, default="MCTS.py",
                   help="MCTS runner script (default: MCTS.py)")
    p.add_argument("--mcts_extra", nargs="*", default=[],
                   help="Extra CLI flags forwarded to the MCTS runner, "
                        "e.g. --cp 0.5")
    p.add_argument("--F3s", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    p.add_argument("--Ts", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--random_indices", nargs="+", default=["0..19"])
    p.add_argument("--script_dir", type=str,
                   default=os.path.join(CODE_ROOT, "experiments",
                                        "section5_questioning_policy"))
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--results_root", type=str,
                   default=os.path.join(CODE_ROOT, "results", "section5", "noise"))
    p.add_argument("--skip_mcts", action="store_true")
    p.add_argument("--skip_heuristics", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--save_timing", action="store_true",
                   help="Propagate --save_timing to each subprocess")
    return p.parse_args()


def main():
    args = parse_args()
    args.random_indices = expand_random_indices(args.random_indices)

    mcts_tasks = [] if args.skip_mcts else build_mcts_tasks(args)
    heur_tasks = [] if args.skip_heuristics else build_heuristic_tasks(args)

    mcts_slots = []
    for g in args.gpus:
        for _ in range(args.mcts_workers_per_gpu):
            mcts_slots.append(g)
    heur_slots = [None] * args.heuristic_workers

    print(f"Planned MCTS tasks: {len(mcts_tasks)} "
          f"across {len(mcts_slots)} workers ({len(args.gpus)} GPUs x "
          f"{args.mcts_workers_per_gpu} per GPU)")
    print(f"Planned heuristic tasks: {len(heur_tasks)} "
          f"across {len(heur_slots)} CPU workers")

    if args.dry_run:
        if mcts_tasks:
            print("sample MCTS:", task_to_cmd(mcts_tasks[0], args.script_dir,
                                              args.python_bin, gpu=args.gpus[0]))
        if heur_tasks:
            print("sample heur:", task_to_cmd(heur_tasks[0], args.script_dir,
                                              args.python_bin))
        return

    t0 = time.time()

    # Run MCTS and heuristic dispatchers in parallel threads so GPUs and
    # CPUs are busy at the same time.
    runners = []
    if mcts_tasks:
        th_mcts = Thread(
            target=dispatch,
            args=(mcts_tasks, mcts_slots, args.script_dir, args.python_bin,
                  args.results_root, "mcts"),
            kwargs={"save_timing": args.save_timing,
                    "mcts_extra": args.mcts_extra},
            daemon=True,
        )
        th_mcts.start()
        runners.append(th_mcts)
    if heur_tasks:
        th_heur = Thread(
            target=dispatch,
            args=(heur_tasks, heur_slots, args.script_dir, args.python_bin,
                  args.results_root, "heur"),
            kwargs={"save_timing": args.save_timing},
            daemon=True,
        )
        th_heur.start()
        runners.append(th_heur)
    for th in runners:
        th.join()

    print(f"[launch_experiments done] total elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
