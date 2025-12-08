import argparse
from pathlib import Path
import os
import tempfile
import subprocess
import shlex
from typing import List

from preprocessing.structural_mri import debias_and_reorient, coregister_images

from utils import log


def batch_debias_and_reorient(input_dir: Path, output_dir: Path, overwrite: bool) -> List[Path]:
   assert input_dir.is_dir()
   output_dir.mkdir(parents=True, exist_ok=True)

   input_files = sorted(input_dir.rglob('*.nii.gz'))
   output_files = []
   for file in input_files:
      relative_path = file.relative_to(input_dir)
      file_output_dir = output_dir / relative_path.parent
      out_file = debias_and_reorient(file, file_output_dir, skip_exist=not overwrite)
      output_files.append(out_file)

   return output_files


def _run_hdbet_batch(input_dir: Path, output_dir: Path) -> None:
    assert input_dir.is_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["hd-bet", "-i", str(input_dir), "-o", str(output_dir), '--save_bet_mask']
    log(f"[BET] running: {' '.join(shlex.quote(c) for c in cmd)}")    # for logging/debugging
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        log(f"[BET][stderr] {result.stderr.strip()}")
        raise RuntimeError("HD-BET failed for batch mode")


def batch_brain_extraction(input_dir: Path, output_dir: Path, overwrite: bool) -> List[Path]:
    assert input_dir.is_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.rglob('*.nii.gz'))

    # Create temp directory with flat structure for batch processing
    with tempfile.TemporaryDirectory(prefix="hdbet_batch_input_") as tmp_in:
        tmp_in = Path(tmp_in)
        with tempfile.TemporaryDirectory(prefix="hdbet_batch_output_") as tmp_out:
            tmp_out = Path(tmp_out)

            # Determine which files need processing
            files_to_process = []
            for file in input_files:
                relative_path = file.relative_to(input_dir)
                out_path = output_dir / relative_path
                if overwrite or not out_path.exists():
                    files_to_process.append((file, relative_path))

            if not files_to_process:
                log(f"  [BET] All {len(input_files)} files already processed, skipping")
                return [output_dir / file.relative_to(input_dir) for file in input_files]

            log(f"  [BET] Processing {len(files_to_process)} files, {len(input_files) - len(files_to_process)} already processed")

            # Create symlinks with unique names in temp input directory
            file_mapping = {}  # Maps temp filename -> (original_file, relative_path)
            for idx, (file, relative_path) in enumerate(files_to_process):
                # Use index to ensure unique names
                temp_name = f"file_{idx:06d}.nii.gz"
                os.symlink(file.resolve(), tmp_in / temp_name)
                file_mapping[temp_name] = (file, relative_path)

            # Run HD-BET in batch mode
            _run_hdbet_batch(tmp_in, tmp_out)

            # Move outputs to correct locations preserving directory structure
            # HD-BET produces two files: the brain-extracted image and the mask (*_bet.nii.gz)
            for temp_name, (original_file, relative_path) in file_mapping.items():
                # Move the brain-extracted image
                temp_output = tmp_out / temp_name
                final_output = output_dir / relative_path
                final_output.parent.mkdir(parents=True, exist_ok=True)

                if temp_output.exists():
                    temp_output.rename(final_output)
                else:
                    raise RuntimeError(f"HD-BET failed to produce output for {original_file}")

                # Move the brain mask file (*_bet.nii.gz)
                temp_mask_name = temp_name.replace('.nii.gz', '_bet.nii.gz')
                temp_mask = tmp_out / temp_mask_name
                if temp_mask.exists():
                    # Create mask path with same relative structure
                    mask_relative_path = Path(str(relative_path).replace('.nii.gz', '_bet.nii.gz'))
                    final_mask = output_dir / mask_relative_path
                    temp_mask.rename(final_mask)

    # Return all expected output files
    expected = [output_dir / file.relative_to(input_dir) for file in input_files]
    if not all(p.exists() for p in expected):
        raise RuntimeError("HD-BET failed for batch mode")

    return expected   # return all brain extracted image files


def batch_mni_reg(input_dir: Path, output_dir: Path, overwrite: bool) -> List[Path]:
    assert input_dir.is_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    fsl_dir = os.getenv('FSLDIR', '/gpfs/share/apps/fsl/6.0.7')
    mni_152_template = Path(fsl_dir) / 'data/standard/MNI152_T1_1mm_brain.nii.gz'


    input_files = sorted(f for f in input_dir.rglob('*.nii.gz') if not str(f).endswith('_bet.nii.gz'))
    output_files = []
    for file in input_files:
        relative_path = file.relative_to(input_dir)
        file_output_dir = output_dir / relative_path.parent
        out_file = coregister_images(mni_152_template, file, file_output_dir, skip_exist=not overwrite)
        output_files.append(out_file)

    return output_files


def preprocess_smri(input_dir: Path, overwrite: bool):
    log(f"Start preprocessing {input_dir}")
    parent_dir = input_dir.parent

    debias_dir = parent_dir / 'debiased'
    batch_debias_and_reorient(input_dir, debias_dir, overwrite=overwrite)

    brain_extract_dir = parent_dir / 'brain_extracted'
    batch_brain_extraction(debias_dir, brain_extract_dir, overwrite=overwrite)

    mni_reg_dir = parent_dir / 'mni_reg'
    batch_mni_reg(brain_extract_dir, mni_reg_dir, overwrite=overwrite)

    log(f"Preprocessing completed for {input_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess structural MRI data')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing the raw data to preprocess')
    parser.add_argument('--overwrite', action='store_true',help='Overwrite existing files')

    args = parser.parse_args()

    preprocess_smri(Path(args.input_dir), args.overwrite)

if __name__ == '__main__':
    main()
