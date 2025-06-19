import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tools
import os
from scipy import ndimage
from simulator import runForwardSolver
import argparse
from pathlib import Path
from parsing import LongitudinalDataset


#%% run LMI WM is needed for registration
def runLMI(registrationReference, patientFlair, patientT1, registrationMode = "WM"):
    atlasPath = "./Atlasfiles"

    wmTransformed, transformedTumor, registration = tools.getAtlasSpaceLMI_InputArray(registrationReference, patientFlair, patientT1, atlasPath, getAlsoWMTrafo=True)

    #%% get the LMI prediction
    prediction = np.array(tools.getNetworkPrediction(transformedTumor))[:6]

    #%% plot the prediction
    D = prediction[0]
    rho = prediction[1]
    T = prediction[2]
    x = prediction[3]
    y = prediction[4]
    z = prediction[5]

    parameterDir = {'D': prediction[0], 'rho': prediction[1], 'T': prediction[2], 'x': prediction[3], 'y': prediction[4], 'z': prediction[5]}

    # run model with the given parameters
    brainPath = os.path.abspath('./simulator/brain')
    absPath = os.path.abspath(atlasPath + '/anatomy_dat/') + '/' # the "/"c is crucial for the solver
    tumor = runForwardSolver.run(absPath, prediction, brainPath)
    np.save('tumor.npy', tumor)

    # register back to patient space
    predictedTumorPatientSpace = tools.convertTumorToPatientSpace(tumor, registrationReference, registration)
    referenceBackTransformed = tools.convertTumorToPatientSpace(wmTransformed, registrationReference, registration)

    return predictedTumorPatientSpace, parameterDir, referenceBackTransformed


if __name__ == "__main__":
    # Example:
    # python infer_dataset.py -cuda_device 0
    # nohup python -u infer_dataset.py -dataset gliodil -cuda_device 4 > tmp_gliodil.out 2>&1 &
    # nohup python -u infer_dataset.py -dataset lumiere -cuda_device 4 > tmp_lumiere.out 2>&1 &
    # nohup python -u infer_dataset.py -dataset rhuh -cuda_device 4 > tmp_rhuh.out 2>&1 &
    # nohup python -u infer_dataset.py -dataset upenn -cuda_device 4 > tmp_upenn.out 2>&1 &
    # nohup python -u infer_dataset.py -dataset ivygap -cuda_device 4 > tmp_ivygap.out 2>&1 &
    # nohup python -u infer_dataset.py -dataset tcga_gbm -cuda_device 4 > tmp_tcga_gbm.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument("-dataset", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    dataset = None
    if args.dataset == "rhuh":
        RHUH_GBM_DIR = Path("/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/rhuh.json")
        rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
        dataset = LongitudinalDataset(dataset_id="RHUH", root_dir=rhuh_root)
        dataset.load(RHUH_GBM_DIR)
    elif args.dataset == "upenn":
        UPENN_GBM_DIR = Path("/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/upenn_gbm.json")
        upenn_gbm_root = "/home/home/lucas/data/UPENN-GBM/UPENN-GBM"
        dataset = LongitudinalDataset(dataset_id="UPENN_GBM", root_dir=upenn_gbm_root)
        dataset.load(UPENN_GBM_DIR)
    elif args.dataset == "gliodil":
        GLIODIL_DIR = Path("/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/gliodil.json")
        gliodil_root = "/mnt/Drive2/lucas/datasets/GLIODIL"
        dataset = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
        dataset.load(GLIODIL_DIR)
    elif args.dataset == "lumiere":
        LUMIERE_DIR = Path("/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/lumiere.json")
        lumiere_root = "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging"
        dataset = LongitudinalDataset(dataset_id="LUMIERE", root_dir=lumiere_root)
        dataset.load(LUMIERE_DIR)
    elif args.dataset == "ivygap":
        IVYGAP_DIR = Path("/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/ivygap.json")
        ivygap_root = "/mnt/Drive2/lucas/datasets/IVYGAP"
        dataset = LongitudinalDataset(dataset_id="IVYGAP", root_dir=ivygap_root)
        dataset.load(IVYGAP_DIR)
    elif args.dataset == "tcga_gbm":
        TCGA_GBM_DIR = Path("/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/tcga_gbm.json")
        tcga_gbm_root = "/mnt/Drive2/lucas/datasets/TCGA-GBM"
        dataset = LongitudinalDataset(dataset_id="TCGA_TBM", root_dir=tcga_gbm_root)
        dataset.load(TCGA_GBM_DIR)
    if dataset is None:
        raise ValueError(f"Dataset {args.dataset} not implemented.")

    for patient_ind, patient in enumerate(dataset.patients):
        print(f"Predicting {patient_ind}/{len(dataset.patients)}...")

        for exam in patient["exams"]:
            if exam["timepoint"] != "preop":
                continue

            if args.dataset == "gliodil":
                patient_dir = exam["t1c"].parent / "preop"
            elif args.dataset == "upenn":
                patient_dir = exam["t1"].parent
            else:
                patient_dir = exam["t1c"].parent
            print(patient_dir)

            wmSegmentationNiiPath = str(patient_dir / "processed/tissue_segmentation/wm_pbmap.nii.gz")
            tumorsegPath = str(patient_dir / "processed/tumor_segmentation/tumor_seg.nii.gz")
            resultPath = str(patient_dir / "processed/growth_models/lmi")
            os.makedirs(resultPath, exist_ok=True)

            try:
                tumorNib = np.rint(nib.load(tumorsegPath).get_fdata()).astype(np.uint8)
                patientFlair = (tumorNib==2).astype(np.uint8)
                patientT1 = ((tumorNib==1) | (tumorNib==3)).astype(np.uint8)

                patientWMNib = nib.load(wmSegmentationNiiPath)
                patientWM = patientWMNib.get_fdata()
                patientWMAffine = patientWMNib.affine

                predictedTumorPatientSpace, parameterDir, wmBackTransformed = runLMI(
                    patientWM,
                    patientFlair,
                    patientT1,
                )

                np.save(os.path.join(resultPath, "lmi_parameters.npy"), parameterDir)
                nib.save(nib.Nifti1Image(predictedTumorPatientSpace, patientWMAffine), os.path.join(resultPath, 'lmi_pred.nii.gz'))
                nib.save(nib.Nifti1Image(wmBackTransformed, patientWMAffine), os.path.join(resultPath, 'lmi_wm_patientSpace.nii.gz'))
            except Exception as e:
                print(f"Exception for {patient_ind}: {e}")
   
    print("Done.")
