## BASIC IMPORTS ##
import os
from os import path as p
import psutil


# Utility Functions
def print_splash() -> None:
    """
    Prints a splash screen to the console when drDeltaG is called.

    Contains a colorful ASCII art of the Greek letter delta (Δ) and a short
    description of the program.
    """
    deepPurple = "\033[35m"
    pink = "\033[95m"
    deltaBar = f"{deepPurple}--{pink}Δ{deepPurple}G"*18
    asciiArt = f"""{deepPurple}

{deepPurple}       ░██        {pink}░███████            ░██   ░██{deepPurple}             ░██████  
{deepPurple}       ░██        {pink}░██   ░██           ░██   ░██{deepPurple}            ░██   ░██ 
{deepPurple} ░████████░██░████{pink}░██    ░██ ░███████ ░██░████████░██████{deepPurple} ░██        
{deepPurple}░██    ░██░███    {pink}░██    ░██░██    ░██░██   ░██        ░██{deepPurple}░██  █████ 
{deepPurple}░██    ░██░██     {pink}░██    ░██░█████████░██   ░██   ░███████{deepPurple}░██     ██ 
{deepPurple}░██   ░███░██     {pink}░██   ░██ ░██       ░██   ░██  ░██   ░██{deepPurple} ░██  ░███ 
{deepPurple} ░█████░██░██     {pink}░███████   ░███████ ░██    ░████░█████░██{deepPurple} ░█████░█ 
{deepPurple}                                                                 
    """
    text = "\tIn-Place Docking with GNINA for MD Trajectories"
    noColor = "\033[0m"
    print(deepPurple + deltaBar + noColor)
    print(asciiArt + noColor)
    print(deepPurple + text + noColor)
    print(deepPurple + deltaBar + noColor)

def toggle_cuda(mode: str, cuda_devices: str = None) -> str | None:
    """
    Toggle CUDA visibility by setting CUDA_VISIBLE_DEVICES environment variable.

    Args:
        mode (str): "ON" to enable CUDA, "OFF" to disable it.
        cuda_devices (str, optional): Original CUDA devices to restore when enabling.

    Returns:
        str | None: Original CUDA_VISIBLE_DEVICES value if disabling, None if enabling.
    """
    if mode == "OFF":
        originalCudaDevices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return originalCudaDevices
    elif mode == "ON":
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        return None

def get_available_memory() -> int:
    """
    Get available system memory in bytes.

    Returns:
        int: Available memory in bytes.
    """
    return psutil.virtual_memory().available

def init_tqdm_bar_options() -> dict[str, str | int | bool]:
    """
    Initialize options for the tqdm progress bar.

    Returns:
        dict: Configuration options for tqdm.
    """
    deepPurple = "\033[95m"
    resetTextColor = "\033[0m"
    tqdmBarOptions = {
        "desc": f"{deepPurple}Re-Evaluating Frames{resetTextColor}",
        "ascii": "->O",
        "colour": "MAGENTA",
        "unit": "frame",
        "ncols": 72,
        "dynamic_ncols": False
    }
    return tqdmBarOptions

def parse_gnina_log(gnina_log: str) -> float:
    """
    Parse the GNINA log file to extract the binding affinity.

    Args:
        gnina_log (str): Path to the GNINA log file.

    Returns:
        float: Binding affinity value extracted from the log.
    """
    with open(gnina_log, "r") as f:
        for line in f:
            if not line.startswith("Affinity:"):
                continue
            return float(line.split()[1])
