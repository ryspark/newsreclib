import os
os.chdir("../")
cmd = "sbatch evals/{script} {pseudocount} {method}"

for script in os.listdir("evals"):
    if not script.endswith(".sh"):
        continue
    for pseudocount in [50]:#[0, 1, 5, 10, 100]:
        for method in ['embed', 'resample', 'base']:
            if method == 'base' and pseudocount != 0 or pseudocount == 0 and method != 'base':
                continue
            os.system(cmd.format(script=script, pseudocount=pseudocount, method=method))
