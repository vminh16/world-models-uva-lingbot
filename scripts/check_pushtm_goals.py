# Reproduce PushTImageRunner's env setup (fix_goal=False) and read the goal pose of every test seed (ISSUE-005).
# Usage, from uva/ with PYTHONPATH=. : python ../scripts/check_pushtm_goals.py <global_seed> <n_envs> [seeded]
import sys, math, numpy as np, random
from unified_video_action.env_runner.pusht_image_runner import PushTImageRunner

seed, n_envs = int(sys.argv[1]), int(sys.argv[2])
seeded_goal = len(sys.argv) > 3 and sys.argv[3] == "seeded"
np.random.seed(seed); random.seed(seed)
r = PushTImageRunner(output_dir="/tmp/goalcheck", n_train=0, n_train_vis=0, n_test=50, n_test_vis=0,
                     legacy_test=True, test_start_seed=100000, max_steps=300, n_obs_steps=16,
                     n_action_steps=8, n_envs=n_envs, fix_goal=False,
                     seeded_goal=seeded_goal)
env = r.env
goals = []
for c in range(math.ceil(50 / n_envs)):
    fns = r.env_init_fn_dills[c * n_envs:(c + 1) * n_envs]
    fns = fns + [r.env_init_fn_dills[0]] * (n_envs - len(fns))
    env.call_each("run_dill_function", args_list=[(x,) for x in fns])
    env.reset()
    g = env.call("get_attr", "goal_pose")
    goals += [tuple(np.round(x, 1)) for x in g][: min(n_envs, 50 - c * n_envs)]
env.close()
print(f"seed={seed} n_envs={n_envs} seeded_goal={seeded_goal} distinct goals among 50 test seeds: {len(set(goals))}")
for i in [0, 1, 2, 9, 10, 11, 49]:
    print(f"  test seed {100000+i}: goal (x, y, theta) = {goals[i]}")
