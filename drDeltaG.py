## BASIC IMPORTS ##
import os
from os import path as p
from subprocess import call, PIPE
import pandas as pd
import argparse
import shutil
import numpy as np
## MULTIPROCESSING IMPORTS ##
import multiprocessing as mp
from tqdm import tqdm
## PDB UTILS IMPORTS ##
from pdbUtils.pdbUtils import pdb2df, df2pdb
## MDA IMPORTS ##
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.transformations import unwrap, center_in_box, fit_rot_trans, nojump
import warnings

## drDeltaG IMPORTS ##
import ddgUtils

# Suppress specific MDAnalysis warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.coordinates.DCD")
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis.coordinates.PDB")
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis.transformations.nojump")


###################################################################
# Argument Parsing
def parse_arguments() -> tuple[str, str, str, int, str, int, str]:
    """
    Parse command-line arguments for the drDeltaG script.

    Returns:
        tuple: (pdb_file, dcd_file, ligand_name, frequency, working_dir, cpus, out_csv)
            - pdb_file (str): Path to the PDB file.
            - dcd_file (str): Path to the DCD file.
            - ligand_name (str): Name of the ligand.
            - frequency (int): Frame sampling frequency.
            - working_dir (str): Directory for temporary files.
            - cpus (int): Number of CPUs to use.
            - out_csv (str): Path to output CSV file.
    """
    parser = argparse.ArgumentParser(description="Parser for drDeltaG script inputs")
    parser.add_argument(
        "--out_csv", "-o",
        type=str,
        default=p.join(os.getcwd(), "affinity_over_time.csv"),
        help="Output CSV (default: ./affinity_over_time.csv)"
    )
    parser.add_argument(
        "--pdb", "-p",
        type=str,
        default="trajectory.pdb",
        help="PDB file name (default: trajectory.pdb)"
    )
    parser.add_argument(
        "--dcd", "-d",
        type=str,
        default="trajectory.dcd",
        help="DCD file name (default: trajectory.dcd)"
    )
    parser.add_argument(
        "--ligand_name", "-l",
        type=str,
        default="LIG",
        help="Ligand name (default: LIG)"
    )
    parser.add_argument(
        "--frequency", "-f",
        type=int,
        default=10,
        help="Frequency value (default: 10)"
    )
    parser.add_argument(
        "--working_dir", "-wd",
        type=str,
        default=os.getcwd(),
        help="Working Directory to make temporary files"
    )
    parser.add_argument(
        "--cpus", "-c",
        type=int,
        default=1,
        help="Number of CPUs to use [WARNING, this can be memory intensive] (default: 1)"
    )
    args = parser.parse_args()
    return args.pdb, args.dcd, args.ligand_name, args.frequency, args.working_dir, args.cpus, args.out_csv
###################################################################
# Trajectory and PDB Handling
def load_trajectory_and_pdb(pdb_file: str, dcd_file: str, ligandName: str) -> mda.Universe:
    """
    Load an MDAnalysis Universe from PDB and DCD files.

    Args:
        pdb_file (str): Path to the PDB file.
        dcd_file (str): Path to the DCD file.

    Returns:
        mda.Universe: Loaded Universe object containing the trajectory.
    """
    inputUniverse = mda.Universe(pdb_file, dcd_file)
    
    # 1. Select the atoms of interest first
    atomSelection = inputUniverse.select_atoms(f"protein or resname {ligandName}")

    # 2. Iterate through the trajectory to collect data from VALID frames
    good_coords = []
    good_dims = []
    frames_skipped = 0

    for timeStep in inputUniverse.trajectory:
        # The key validation step: check if dimensions are valid
        if timeStep.dimensions is not None and np.all(np.isfinite(timeStep.dimensions)):
            # If valid, store the coordinates of our selection and the box dimensions
            good_coords.append(atomSelection.positions.copy())
            good_dims.append(timeStep.dimensions.copy())
        else:
            frames_skipped += 1
    if frames_skipped > 0:
        raise ValueError(f"Frames skipped due to bad PBC Dimensions: {frames_skipped}")

    # 3. Create a new Universe from the selection
    # The topology (atoms, bonds, etc.) is taken from the selection
    trimmedUniverse = mda.Merge(atomSelection)

    # 4. Assign the collected "good" data to the new universe's trajectory
    # Convert the list of coordinates and dimensions into 3D numpy arrays
    # This creates an in-memory trajectory (MemoryReader)
    trimmedUniverse.load_new(
        np.array(good_coords),       # Coordinates must be a (n_frames, n_atoms, 3) array
        dimensions=np.array(good_dims) # Dimensions must be a (n_frames, 6) array
    )

    return trimmedUniverse

###################################################################
@ddgUtils.bouncing_bar_decorator(ball_text=".oO Aligning Trajectory Oo.")
def align_structure(u: mda.Universe, ligandName: str) -> mda.Universe:
    """
    Aligns a trajectory, correctly handling Periodic Boundary Conditions (PBC).

    The workflow first makes all molecules whole, then centers the protein-heme
    complex, and finally aligns the trajectory to the first frame.
    """
    ## we need bond data for Cl- and Na+ (why? not sure!)
    u = add_bond_data(u)
    ## selections
    complex_selection = u.select_atoms(f"protein or resname {ligandName}")
    alignment_selection = u.select_atoms(f"protein and name CA")

    workflow = [
        nojump.NoJump(complex_selection),                       ## stops things from jumping
        unwrap(u.atoms),                                        ## unwrap to remove PBC
        center_in_box(complex_selection, wrap=True),            ## center complex, re-wrap
        fit_rot_trans(alignment_selection, alignment_selection) ## remove translation and rotation
    ]
    
    u.trajectory.add_transformations(*workflow)
    return u
###################################################################
def write_pdb_for_frame(args):
    """
    Worker function to write a PDB file for a single frame.
    
    Args:
        args (tuple): Contains (universe, frame_idx, output_dir, prefix, selection).
    
    Returns:
        tuple: (frame_id as string, output PDB file path).
    """
    universe, frame_idx, output_dir, prefix, selection = args
    atoms = universe.select_atoms(selection)
    universe.trajectory[frame_idx]
    output_file = os.path.join(output_dir, f"{prefix}_{(frame_idx+1):04d}.pdb")
    atoms.write(output_file)
    return (f"{(frame_idx+1):04d}", output_file)
###################################################################
@ddgUtils.bouncing_bar_decorator(bar_width=72, ball_text=".oO Splitting Trajectory Oo.")
def split_universe_to_pdbs(universe: mda.Universe, frequency: int = 1, output_dir: str = "pdb_frames", prefix: str = "frame", selection: str = "all", num_processes: int = None) -> dict[str, str]:
    """
    Split an MDAnalysis Universe trajectory into separate PDB files for each frame using multiprocessing.

    Args:
        universe (Universe): The MDAnalysis Universe object with the trajectory.
        frequency (int): Frequency of frames to process (default: 1).
        output_dir (str): Directory to save PDB files (default: 'pdb_frames').
        prefix (str): Prefix for output PDB filenames (default: 'frame').
        selection (str): Atom selection string (default: 'all').
        num_processes (int): Number of processes to use. If None, uses all available CPUs.

    Returns:
        dict: Mapping of frame indices (as strings) to PDB file paths.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Default to all available CPU cores if num_processes is not specified
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # List of frame indices to process, respecting the frequency
    frame_indices = [i for i in range(len(universe.trajectory)) if i % frequency == 0]
    
    # Prepare arguments for each worker: a tuple of (universe, frame_idx, output_dir, prefix, selection)
    args_list = [(universe, frame_idx, output_dir, prefix, selection) for frame_idx in frame_indices]
    
    # Use a multiprocessing pool to process frames in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(write_pdb_for_frame, args_list)
    
    # Convert list of (frame_id, file_path) tuples into a dictionary
    frame_pdbs = dict(results)
    return frame_pdbs
###################################################################
def split_pdb_file(pdb_file: str, ligand_name: str, frame_id: int, out_dir: str) -> tuple[str, str]:
    """
    Split a PDB file into protein and ligand parts based on ligand name.

    Args:
        pdb_file (str): Path to the input PDB file.
        ligand_name (str): Residue name of the ligand.
        frame_id (int): Frame identifier for output file naming.
        out_dir (str): Directory to save the split PDB files.

    Returns:
        tuple: (prot_pdb, lig_pdb)
            - prot_pdb (str): Path to the protein PDB file.
            - lig_pdb (str): Path to the ligand PDB file.
    """
    pdbDf = pdb2df(pdb_file)
    protDf = pdbDf[pdbDf["RES_NAME"] != ligand_name]
    ligDf = pdbDf[pdbDf["RES_NAME"] == ligand_name]
    protPdb = p.join(out_dir, f"frame_{frame_id}_prot.pdb")
    ligPdb = p.join(out_dir, f"frame_{frame_id}_lig.pdb")
    df2pdb(protDf, protPdb)
    df2pdb(ligDf, ligPdb)
    return protPdb, ligPdb
###################################################################
# GNINA Docking Functions
def run_gnina_inplace_docking(prot_pdb: str, lig_pdb: str, frame_idx: int, out_dir: str, mem_fraction: float = 0.8) -> str:
    """
    Run GNINA docking with a memory limit based on available system memory.

    Args:
        prot_pdb (str): Path to the protein PDB file.
        lig_pdb (str): Path to the ligand PDB file.
        frame_idx (int): Frame index for logging and file naming.
        out_dir (str): Directory for output files.
        mem_fraction (float): Fraction of available memory to allocate (default: 0.8).

    Returns:
        str: Path to the GNINA log file.
    """
    gninaLog = p.join(out_dir, f"frame_{frame_idx}_gnina.log")
    gninaCommand = [
        "gnina",
        "-r", prot_pdb,
        "-l", lig_pdb,
        "--score_only",
        "--log", gninaLog,
        "--cpu", "1",
        "--no_gpu",
        "--scoring", "vina",
        "--cnn_scoring", "none",
    ]
    call(gninaCommand, stdout=PIPE, stderr=PIPE)
    return gninaLog

###################################################################
def process_frame_pdb(pdb_file: str, ligand_name: str, frame_idx: int, out_dir: str, debug: bool = False) -> float:
    """
    Process a single frame PDB: split into protein/ligand, run GNINA, and get affinity.

    Args:
        pdb_file (str): Path to the frame PDB file.
        ligand_name (str): Name of the ligand residue.
        frame_idx (int): Frame index.
        out_dir (str): Directory for output files.
        debug (bool): If True, keep temporary files; if False, delete them (default: False).

    Returns:
        float: Binding affinity from GNINA docking.
    """
    protPdb, ligPdb = split_pdb_file(pdb_file, ligand_name, frame_idx, out_dir)
    gninaLog = run_gnina_inplace_docking(protPdb, ligPdb, frame_idx, out_dir)
    bindingAffinity = ddgUtils.parse_gnina_log(gninaLog)
    if not debug:
        [os.remove(file) for file in [protPdb, ligPdb]]
    return bindingAffinity
###################################################################
# Parallel Processing
def worker(args: tuple[str, str, str, str]) -> tuple[str, float]:
    """
    Worker function to process a single frame PDB and return binding affinity.

    Args:
        args (tuple): (frame_idx, frame_pdb, ligand_name, split_pdb_dir)
            - frame_idx (str): Frame index.
            - frame_pdb (str): Path to frame PDB file.
            - ligand_name (str): Name of the ligand.
            - split_pdb_dir (str): Directory for split PDB files.

    Returns:
        tuple: (frame_idx, binding_affinity)
            - frame_idx (str): Frame index.
            - binding_affinity (float): Binding affinity value.
    """
    frameIdx, framePdb, ligandName, splitPdbDir, debug = args
    bindingAffinity = process_frame_pdb(framePdb, ligandName, frameIdx, splitPdbDir, debug)
    return frameIdx, bindingAffinity
###################################################################
def parallel_process_frame_pdbs(frame_pdbs: dict[str, str], ligand_name: str, split_pdb_dir: str, num_processes: int | None = None, debug: bool = False) -> dict[str, float]:
    """
    Process frame PDBs in parallel and return affinity over time.

    Args:
        frame_pdbs (dict): Mapping of frame indices to PDB file paths.
        ligand_name (str): Name of the ligand.
        split_pdb_dir (str): Directory for split PDB files.
        num_processes (int | None): Number of processes; if None, uses CPU count (default: None).

    Returns:
        dict: Mapping of frame indices to binding affinities.
    """
    if num_processes is None:
        numProcesses = mp.cpu_count()
    else:
        numProcesses = num_processes
    affinityOverTime = {}
    argsList = [(frame_idx, frame_pdb, ligand_name, split_pdb_dir, debug) 
                for frame_idx, frame_pdb in frame_pdbs.items()]
    with mp.Pool(processes=numProcesses) as pool:
        results = list(tqdm(pool.imap(worker, argsList), total=len(argsList), **ddgUtils.init_tqdm_bar_options()))
        for frameIdx, bindingAffinity in results:
            affinityOverTime[frameIdx] = bindingAffinity
    return affinityOverTime
@ddgUtils.bouncing_bar_decorator(bar_width=72, ball_text="Oo.Writing DCD.oO")
def write_dcd(u: mda.Universe, output_file: str):
    """
    Write the trajectory to a DCD file.

    Args:
        u (mda.Universe): Universe object containing the trajectory.
        output_file (str): Path to the output DCD file.
    """
    atom_selection = u.select_atoms("protein or resname HEM")  # Replace with your selection, or use u.atoms for all atoms

    with mda.Writer(output_file, n_atoms=atom_selection.n_atoms) as dcd_writer:
        # Iterate over the trajectory
        for ts in u.trajectory:
            # Write the current frame for the selected atoms
            dcd_writer.write(atom_selection)
###################################################################
def add_bond_data(u):
    # --- THIS IS THE FIX ---
    # Create a dictionary to hold the custom vdW radii.
    # The key is the atom type name ('Na') from the error message.
    # The value is the radius in Angstroms.
    custom_radii = {'Na': 2.2, 
                    'Cl': 2.2 } # Using a standard value for Sodium

    # Now, call guess_bonds() and pass your custom radii.
    # MDAnalysis will use this value for 'Na' atoms and its internal
    # defaults for all other atom types.
    u.atoms.guess_bonds(vdwradii=custom_radii)

    return u
###################################################################
def main(debug=True) -> None:
    """
    Main function to run the drDeltaG script.

    Executes the full workflow: parsing arguments, loading trajectory, splitting frames,
    running docking in parallel, and saving results.
    """
    ddgUtils.print_splash()
    cudaDevices = ddgUtils.toggle_cuda("OFF")
    pdbFile, dcdFile, ligandName, frequency, workingDir, numCpus, outCsv = parse_arguments()
    os.makedirs(workingDir, exist_ok=True)

    
    unalignedUniverse = load_trajectory_and_pdb(pdbFile, dcdFile, ligandName)
    alignedUniverse = align_structure(unalignedUniverse, ligandName)
    del unalignedUniverse

    if debug:
        alignedDcd = p.join(workingDir, "aligned.dcd")
        write_dcd(alignedUniverse, alignedDcd)

    pdbFrameDir = p.join(workingDir, "TMP_PDB_FRAMES")
    framePdbs = split_universe_to_pdbs(alignedUniverse, output_dir=pdbFrameDir, frequency=frequency, prefix="frame", selection='not (resname HOH WAT TIP3 SPC NA CL)', num_processes=numCpus)
    splitPdbDir = p.join(workingDir, "TMP_SPLIT_PDBS")
    os.makedirs(splitPdbDir, exist_ok=True)
    affinityOverTime = {}
    affinityOverTime = parallel_process_frame_pdbs(framePdbs, ligandName, splitPdbDir, num_processes=numCpus, debug=debug)
    affinityDf = pd.DataFrame.from_dict(affinityOverTime, orient="index")
    affinityDf.to_csv(outCsv)
    ddgUtils.toggle_cuda("ON", cudaDevices)
    if not debug:
        clean_up(workingDir)
###################################################################
def clean_up(working_dir: str) -> None:
    """
    Clean up temporary directories created during execution.

    Args:
        working_dir (str): Directory containing temporary folders to remove.
    """
    shutil.rmtree(p.join(working_dir, "TMP_PDB_FRAMES"))
    shutil.rmtree(p.join(working_dir, "TMP_SPLIT_PDBS"))
###################################################################
if __name__ == "__main__":
    main()