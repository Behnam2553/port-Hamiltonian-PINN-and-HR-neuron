import subprocess, time, sys, pathlib

CMD = [sys.executable, "v_dot_parameter_sweep_2D.py"]
# CMD = [r"C:\Virtual Environments\Neuron\.venv\Scripts\python.exe", "v_dot_parameter_sweep_2D.py"]


while True:
    print("\n=== Starting simulation at", time.ctime())
    exit_code = subprocess.call(CMD)
    if exit_code == 0:
        print("=== Simulation finished successfully!")
        break
    print(f"*** Child exited with code {exit_code}.  Restarting in 15 s …")
    time.sleep(5)
