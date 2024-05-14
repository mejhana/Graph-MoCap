from subprocess import run
from pathlib import Path

results_dir = Path.cwd() / "results"
loopable_results_dir = Path.cwd() / "loopable_results"

for gif in results_dir.glob("*.gif"):
    # print(gif)
    args = ["convert", "-loop", "0", gif, loopable_results_dir / gif.name]
    # print(" ".join(map(str, args)))
    print(run(args, capture_output=True))
