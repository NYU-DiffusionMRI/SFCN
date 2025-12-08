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

   input_files = sorted(input_dir.glob('*.nii.gz'))
   output_files = []
   for file in input_files:
      out_file = debias_and_reorient(file, output_dir, skip_exist=not overwrite)
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

    input_files = sorted(input_dir.glob('*.nii.gz'))

    if overwrite:
        # simple HD-BET with batch support
        log(f"  [BET] In Batch Mode (overwrite), {len(list(input_dir.glob('*.nii.gz')))} files to process")

        _run_hdbet_batch(input_dir, output_dir)
    else:
        missing_files = [p for p in input_files if not (output_dir / p.name).exists()]
        existing_files = [p for p in input_files if (output_dir / p.name).exists()]

        log(f"  [BET] In Batch Mode (no overwrite), {len(missing_files)} files to process, {len(existing_files)} files already processed")
        if missing_files:
            if len(missing_files) == len(input_files):
                # Fast path: identical to overwrite
                _run_hdbet_batch(input_dir, output_dir)
            else:
                with tempfile.TemporaryDirectory(prefix="hdbet_batch_") as tmp:
                    tmp = Path(tmp)
                    for p in missing_files:
                        os.symlink(p.resolve(), tmp / p.name)

                    _run_hdbet_batch(tmp, output_dir)

    expected = [output_dir / p.name for p in input_files]
    if not all(p.exists() for p in expected):
        raise RuntimeError("HD-BET failed for batch mode")

    return expected   # return all brain extracted image files


def batch_mni_reg(input_dir: Path, output_dir: Path, overwrite: bool) -> List[Path]:
    assert input_dir.is_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    fsl_dir = os.getenv('FSLDIR', '/gpfs/share/apps/fsl/6.0.7')
    mni_152_template = Path(fsl_dir) / 'data/standard/MNI152_T1_1mm_brain.nii.gz'


    input_files = sorted(f for f in input_dir.glob('*.nii.gz') if not str(f).endswith('_bet.nii.gz'))
    output_files = []
    for file in input_files:
        out_file = coregister_images(mni_152_template, file, output_dir, skip_exist=not overwrite)
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
