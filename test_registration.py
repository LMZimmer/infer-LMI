import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

import tools


def run_registration_identity(patient_wm_path: Path, tumor_seg_path: Path, out_dir: Path) -> None:
    """Runs the registration steps of the LMI pipeline but returns a downsampled
    version of the input tumor segmentation as prediction output. This checks
    whether the forward and inverse transformations are performed correctly."""
    atlas_path = Path(__file__).resolve().parent / "Atlasfiles"

    # Load input volumes
    patient_wm_nii = nib.load(str(patient_wm_path))
    patient_wm = patient_wm_nii.get_fdata()
    patient_affine = patient_wm_nii.affine

    tumor_nii = nib.load(str(tumor_seg_path))
    tumor_data = np.rint(tumor_nii.get_fdata()).astype(np.uint8)

    # Create modality specific masks as done during inference
    patient_flair = (tumor_data == 2).astype(np.uint8)
    patient_t1 = ((tumor_data == 1) | (tumor_data == 3)).astype(np.uint8)

    # Registration to atlas space
    wm_transformed, tumor_transformed, registration = tools.getAtlasSpaceLMI_InputArray(
        patient_wm,
        patient_flair,
        patient_t1,
        str(atlas_path),
        getAlsoWMTrafo=True,
        patientAffine=patient_affine,
    )

    # Downsample tumor in atlas space to match network prediction size
    factors = np.array([128, 128, 128]) / np.array(tumor_transformed.shape)
    tumor_down = zoom(tumor_transformed, factors, order=0)

    # Instead of running the solver, directly convert the downsampled tumor back
    tumor_back = tools.convertTumorToPatientSpace(
        tumor_down, patient_wm, registration, patientAffine=patient_affine
    )

    wm_back = tools.convertTumorToPatientSpace(
        wm_transformed, patient_wm, registration, patientAffine=patient_affine
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(tumor_back, patient_affine), out_dir / "tumor_registered.nii.gz")
    nib.save(nib.Nifti1Image(wm_back, patient_affine), out_dir / "wm_registered.nii.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test registration by mapping tumour to atlas and back")
    parser.add_argument("--wm", required=True, type=Path, help="Path to patient WM segmentation")
    parser.add_argument("--tumor", required=True, type=Path, help="Path to patient tumor segmentation")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for test results")
    args = parser.parse_args()

    run_registration_identity(args.wm, args.tumor, args.out)
