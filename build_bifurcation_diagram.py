import numpy as np
import subprocess

verbose = False
Ras = list(np.geomspace(1e3, 1e5, num=11))
Rac = 1708
Rac_bracket = [1.7e3, 1.75e3, 1.8e3, 1.85e3, 1.9e3]
Ras += Rac_bracket
for Ra in Ras:
    stop_time = 1e2
    if Ra < Rac_bracket[0]:
        Nz = 8
    elif (Ra >= Rac_bracket[0] and Ra <= Rac_bracket[-1]):
        Nz = 16
        stop_time = 2e3
    elif Ra <= 6e4:
        Nz = 32
    else:
        Nz = 64
    aspect = 8
    command = f'mpirun -np 4 python3 rayleigh_benard_2.5d.py --Nz={Nz:d} --run_time_sim={stop_time} --Rayleigh={Ra:.3e} --aspect={aspect:d}'
    print(f'command: {command:s}')
    proc = subprocess.run(command,
                          shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.stdout, proc.stderr

    if verbose:
        for line in stdout.splitlines():
            print("out: {}".format(line))

        for line in stderr.splitlines():
            print("err: {}".format(line))
