# drDeltaG - Binding Affinity Analysis for MD Trajectories

## Introduction

drDeltaG is a Python script designed to analyze molecular dynamics (MD) trajectories by calculating the binding affinity of a ligand to a protein over time. It leverages GNINA, a deep learning-based docking tool, to score the ligand's binding pose in each frame of the trajectory. The script is optimized for performance, utilizing multiprocessing to handle large trajectories efficiently.

## Prerequisites

To run drDeltaG, ensure you have the following installed:

- **Python 3.6 or higher**
- **MDAnalysis** for trajectory analysis
- **GNINA** for docking and scoring
- Additional Python packages: `pandas`, `pdbUtils`, `psutil`, `tqdm`

### Installation

Install the required Python packages using pip:

```bash
pip install MDAnalysis pandas psutil tqdm pdbutils
```

Download the `pre-built binary` of GNINA from [https://github.com/gnina/gnina](https://github.com/gnina/gnina), and ensure it is added to your system's PATH.

## Usage

Run the script from the command line with the appropriate arguments:

```bash
python drDeltaG.py [options]
```

### Available Options

- `--out_csv, -o`: Path to the output CSV file (default: `./affinity_over_time.csv`)
- `--pdb, -p`: Path to the PDB file (default: `trajectory.pdb`)
- `--dcd, -d`: Path to the DCD trajectory file (default: `trajectory.dcd`)
- `--ligand_name, -l`: Residue name of the ligand (default: `LIG`)
- `--frequency, -f`: Frequency of frames to process (default: `10`)
- `--working_dir, -wd`: Directory for temporary files (default: current working directory)
- `--cpus, -c`: Number of CPUs to use (default: `1`)

### Example

```bash
python drDeltaG.py --pdb protein.pdb --dcd trajectory.dcd --ligand_name LIG --frequency 5 --cpus 4 --out_csv results.csv
```

This command processes every 5th frame of the trajectory using 4 CPUs and saves the binding affinities to `results.csv`.

## Output

The script generates a CSV file with two columns:

- **Frame index** (e.g., `0001`, `0002`, etc.)
- **Binding affinity** in kcal/mol

The frame indices correspond to the frames processed based on the specified frequency.

## Notes

- The script assumes the ligand is present in each frame. If not, the splitting step may fail.
- The atom selection excludes common solvent and ion residues (`HOH`, `WAT`, `TIP3`, `SPC`, `NA`, `CL`). Adjust the selection in the code if your system uses different names.
- Temporary directories (`TMP_PDB_FRAMES` and `TMP_SPLIT_PDBS`) are removed after processing. To keep these files for debugging, modify the `clean_up` function.