from pathlib import Path
import subprocess
from typing import List
import ants

from utils import log



def get_basename(p: Path) -> str:
    name = p.name
    if name.lower().endswith('.nii.gz'):
        return name[:-7]
    if name.lower().endswith('.nii'):
        return name[:-4]
    return p.stem


def _run_hdbet(input_file: Path, out_dir: Path, device: str='cuda'):
    """Run HD-BET on a single file, writing to out_dir/{basename}.nii.gz and _mask.nii.gz"""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = get_basename(input_file)
    out_file = out_dir / input_file.name

    log(f"  [BET] running on {base}")
    result = subprocess.run(
        ['hd-bet', '-i', str(input_file), '-o', str(out_file), '--save_bet_mask', '-device', device],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        log(f"  [BET][stderr] {result.stderr.strip()}")
        raise RuntimeError(f"HD-BET failed for {base}")



def brain_extraction(fixed_img: Path, out_dir: Path, skip_exist=False, device: str='cuda'):
    """
    Run HD-BET on fixed_img and return (skullstripped, mask).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = get_basename(fixed_img)
    brain_img = out_dir / f"{base}.nii.gz"
    brain_mask  = out_dir / f"{base}_bet.nii.gz"

    if skip_exist and brain_img.exists() and brain_mask.exists():
        log(f"  [BET] skip existing {base}")
        return brain_img, brain_mask

    _run_hdbet(fixed_img, out_dir, device=device)

    if not brain_img.exists() or not brain_mask.exists():
        raise RuntimeError(f"HD-BET failed for {base}")
    return brain_img, brain_mask


def debias_and_reorient(src: Path, out_dir: Path, skip_exist=False) -> Path:
    """
    N4 bias correction + save .nii.gz
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = get_basename(src)
    out_file = out_dir / f"{base}.nii.gz"
    if skip_exist and out_file.exists():
        log(f"  [N4] skip existing {base}")
        return out_file

    img = ants.image_read(str(src))
    log(f"  [N4] running on {base}")
    img = ants.n4_bias_field_correction(img)
    img = ants.reorient_image2(img, orientation='LPI')
    ants.image_write(img, str(out_file))
    return out_file


def coregister_images(fixed: Path, moving: Path, out_dir: Path, skip_exist=False) -> Path:
    """Affine coregistration of moving to fixed"""

    out_dir.mkdir(parents=True, exist_ok=True)
    base = get_basename(moving)
    out_file = out_dir / f"{base}.nii.gz"
    if skip_exist and out_file.exists():
        log(f"  [REG] skip existing {base}")
        return out_file

    img_f = ants.image_read(str(fixed))
    img_m = ants.image_read(str(moving))

    orient_f = img_f.get_orientation()
    orient_m = img_m.get_orientation()

    if orient_m != orient_f:
        log(f"  [REG] moving image orientation ({orient_m}) does not match fixed image orientation ({orient_f}), reorienting!")
        img_m = ants.reorient_image2(img_m, orientation=orient_f)

    log(f"  [REG] running on {base}")
    reg = ants.registration(fixed=img_f, moving=img_m, type_of_transform='Affine')
    out = ants.apply_transforms(
        fixed=img_f, moving=img_m,
        transformlist=reg['fwdtransforms']
    )
    ants.image_write(out, str(out_file))

    return out_file
