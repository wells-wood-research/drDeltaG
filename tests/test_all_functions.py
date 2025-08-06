import unittest
import os
import sys
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ddgUtils
import drDeltaG

import io

class TestDdgUtils(unittest.TestCase):
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_print_splash(self, mock_stdout):
        """Test that the splash screen prints without errors."""
        ddgUtils.print_splash()
        output = mock_stdout.getvalue()
        self.assertIn("In-Place Docking with GNINA for MD Trajectories", output)
        self.assertIn("Δ", output)

    def test_toggle_cuda(self):
        """Test CUDA device visibility toggling."""
        original_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        # Test turning CUDA OFF
        returned_devices = ddgUtils.toggle_cuda("OFF")
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertEqual(returned_devices, original_devices)

        # Test turning CUDA ON
        ddgUtils.toggle_cuda("ON", "0,1")
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "0,1")

        # Restore original state
        if original_devices is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_devices

    @patch('psutil.virtual_memory')
    def test_get_available_memory(self, mock_virtual_memory):
        """Test retrieval of available system memory."""
        mock_virtual_memory.return_value.available = 1024 * 1024 * 100  # 100 MB
        self.assertEqual(ddgUtils.get_available_memory(), 1024 * 1024 * 100)

    def test_init_tqdm_bar_options(self):
        """Test initialization of tqdm progress bar options."""
        options = ddgUtils.init_tqdm_bar_options()
        self.assertIsInstance(options, dict)
        self.assertIn("desc", options)
        self.assertIn("colour", options)
        self.assertEqual(options["unit"], "frame")

    def test_parse_gnina_log(self):
        """Test parsing of GNINA log file to extract affinity."""
        log_content = "Affinity: -7.5 kcal/mol\nOther line\n"
        m = mock_open(read_data=log_content)
        with patch('builtins.open', m):
            affinity = ddgUtils.parse_gnina_log("dummy_log_path.log")
        self.assertEqual(affinity, -7.5)

import MDAnalysis as mda
import shutil
import pandas as pd
from unittest.mock import call

class TestDrDeltaG(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = "test_temp"
        os.makedirs(self.test_dir, exist_ok=True)
        self.pdb_file = "TEST_INPUTS/trajectory.pdb"
        self.dcd_file = "TEST_INPUTS/trajectory.dcd"
        self.ligand_name = "LIG"
        self.universe = mda.Universe(self.pdb_file, self.dcd_file)

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('sys.argv', ['drDeltaG.py', '--pdb', 'test.pdb', '--dcd', 'test.dcd', '-l', 'MYLIG', '-f', '5', '-wd', 'temp_dir', '-c', '4', '-o', 'out.csv'])
    def test_parse_arguments(self):
        """Test argument parsing."""
        pdb, dcd, lig, freq, wd, cpus, out_csv = drDeltaG.parse_arguments()
        self.assertEqual(pdb, "test.pdb")
        self.assertEqual(dcd, "test.dcd")
        self.assertEqual(lig, "MYLIG")
        self.assertEqual(freq, 5)
        self.assertEqual(wd, "temp_dir")
        self.assertEqual(cpus, 4)
        self.assertEqual(out_csv, "out.csv")

    def test_load_trajectory_and_pdb(self):
        """Test loading of PDB and DCD files into an MDAnalysis Universe."""
        u = drDeltaG.load_trajectory_and_pdb(self.pdb_file, self.dcd_file)
        self.assertIsInstance(u, mda.Universe)
        self.assertEqual(len(u.trajectory), 50)

    def test_write_pdb_for_frame(self):
        """Test writing a single PDB file for a specific frame."""
        output_dir = os.path.join(self.test_dir, "pdbs")
        os.makedirs(output_dir, exist_ok=True)
        args = (self.universe, 0, output_dir, "test_frame", "all")
        frame_id, pdb_path = drDeltaG.write_pdb_for_frame(args)
        self.assertEqual(frame_id, "0001")
        self.assertTrue(os.path.exists(pdb_path))
        self.assertGreater(os.path.getsize(pdb_path), 0)

    def test_split_universe_to_pdbs(self):
        """Test splitting a universe into multiple PDB files."""
        output_dir = os.path.join(self.test_dir, "split_pdbs")
        frame_pdbs = drDeltaG.split_universe_to_pdbs(self.universe, frequency=5, output_dir=output_dir, num_processes=1)
        self.assertEqual(len(frame_pdbs), 10)  # 50 frames, freq 5 -> frames 0, 5, ... 45
        self.assertTrue(os.path.exists(frame_pdbs["0001"]))
        self.assertTrue(os.path.exists(frame_pdbs["0006"]))
        self.assertTrue(os.path.exists(frame_pdbs["0046"]))

    def test_split_pdb_file(self):
        """Test splitting a PDB file into protein and ligand."""
        # Create a dummy PDB file for testing
        pdb_content = """
ATOM      1  N   ALA A   1      27.340  34.400  42.280  1.00  0.00           N
ATOM      2  CA  ALA A   1      28.320  34.330  43.200  1.00  0.00           C
HETATM    3  O   LIG B   1      30.000  35.000  44.000  1.00  0.00           O
"""
        pdb_path = os.path.join(self.test_dir, "test.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_content)

        prot_pdb, lig_pdb = drDeltaG.split_pdb_file(pdb_path, "LIG", "0001", self.test_dir)
        self.assertTrue(os.path.exists(prot_pdb))
        self.assertTrue(os.path.exists(lig_pdb))
        with open(prot_pdb, "r") as f:
            self.assertNotIn("LIG", f.read())
        with open(lig_pdb, "r") as f:
            self.assertIn("LIG", f.read())

    @patch('drDeltaG.call')
    def test_run_gnina_inplace_docking(self, mock_call):
        """Test the GNINA docking command execution."""
        log_path = drDeltaG.run_gnina_inplace_docking("prot.pdb", "lig.pdb", "0001", self.test_dir)
        self.assertTrue(log_path.endswith("_gnina.log"))
        expected_command = [
            "gnina", "-r", "prot.pdb", "-l", "lig.pdb",
            "--score_only", "--log", log_path, "--cpu", "1", "--no_gpu",
            "--scoring", "vina", "--cnn_scoring", "none"
        ]
        mock_call.assert_called_once_with(expected_command, stdout=-1, stderr=-1)


    @patch('drDeltaG.split_pdb_file', return_value=("prot.pdb", "lig.pdb"))
    @patch('drDeltaG.run_gnina_inplace_docking', return_value="dummy.log")
    @patch('ddgUtils.parse_gnina_log', return_value=-8.0)
    def test_process_frame_pdb(self, mock_parse, mock_run_gnina, mock_split):
        """Test processing of a single frame PDB."""
        affinity = drDeltaG.process_frame_pdb("frame.pdb", "LIG", "0001", self.test_dir, debug=True)
        self.assertEqual(affinity, -8.0)
        mock_split.assert_called_with("frame.pdb", "LIG", "0001", self.test_dir)
        mock_run_gnina.assert_called_with("prot.pdb", "lig.pdb", "0001", self.test_dir)
        mock_parse.assert_called_with("dummy.log")

    @patch('drDeltaG.process_frame_pdb', return_value= -9.5)
    def test_worker(self, mock_process):
        """Test the worker function for parallel processing."""
        args = ("0001", "frame.pdb", "LIG", self.test_dir)
        frame_idx, affinity = drDeltaG.worker(args)
        self.assertEqual(frame_idx, "0001")
        self.assertEqual(affinity, -9.5)
        mock_process.assert_called_with("frame.pdb", "LIG", "0001", self.test_dir)

    def test_clean_up(self):
        """Test the cleanup of temporary directories."""
        dir1 = os.path.join(self.test_dir, "TMP_PDB_FRAMES")
        dir2 = os.path.join(self.test_dir, "TMP_SPLIT_PDBS")
        os.makedirs(dir1)
        os.makedirs(dir2)
        drDeltaG.clean_up(self.test_dir)
        self.assertFalse(os.path.exists(dir1))
        self.assertFalse(os.path.exists(dir2))

if __name__ == '__main__':
    unittest.main()
