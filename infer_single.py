#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tools
import os
from scipy import ndimage
from simulator import runForwardSolver


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
    # nohup python -u infer_single.py  > test.out 2>&1 &

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    # Paths
    """
    patientPath = "/mnt/Drive2/lucas/datasets/data_GliODIL_essential/data_716"
    wmSegmentationNiiPath = os.path.join(patientPath, "t1_wm.nii.gz")
    tumorsegPath = os.path.join(patientPath, "segm.nii.gz")
    resultPath = os.path.join(patientPath, "lmi")

    os.makedirs(resultPath, exist_ok=True)
    
    # Load tumor core / edema
    tumorNib = np.rint(nib.load(tumorsegPath).get_fdata()).astype(np.uint8)
    patientFlair = (tumorNib==3).astype(np.uint8)
    patientT1 = ((tumorNib==1) | (tumorNib==4)).astype(np.uint8)
    
    patientPath = "/mnt/Drive2/lucas/datasets/GLIODIL/tgm016/preop/preop/processed"
    wmSegmentationNiiPath = os.path.join(patientPath, "tissue_segmentation/wm_pbmap.nii.gz")
    tumorsegPath = os.path.join(patientPath, "tumor_segmentation/tumor_seg.nii.gz")
    resultPath = os.path.join(patientPath, "growth_models/lmi_test")
    """

    patientPath = "/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0024/11-10-2013-NA-Craneo-58463/processed"
    wmSegmentationNiiPath = os.path.join(patientPath, "tissue_segmentation/wm_pbmap.nii.gz")
    tumorsegPath = os.path.join(patientPath, "tumor_segmentation/tumor_seg.nii.gz")
    resultPath = os.path.join(patientPath, "growth_models/lmi_test")

    os.makedirs(resultPath, exist_ok=True)

    # Load tumor core / edema
    tumorNib = np.rint(nib.load(tumorsegPath).get_fdata()).astype(np.uint8)
    patientFlair = (tumorNib==2).astype(np.uint8)
    patientT1 = ((tumorNib==1) | (tumorNib==3)).astype(np.uint8)

    # Load WM, save affine
    patientWMNib = nib.load(wmSegmentationNiiPath)
    patientWM = patientWMNib.get_fdata()	
    patientWMAffine = patientWMNib.affine

    predictedTumorPatientSpace, parameterDir, wmBackTransformed = runLMI(
        patientWM,
        patientFlair,
        patientT1,
    )

    np.save(os.path.join(resultPath, "lmi_parameters.npy"), parameterDir)

    nib.save(nib.Nifti1Image(predictedTumorPatientSpace, patientWMAffine), os.path.join(resultPath, 'lmi_tumor_patientSpace.nii'))

    nib.save(nib.Nifti1Image(wmBackTransformed, patientWMAffine), os.path.join(resultPath, 'lmi_wm_patientSpace.nii'))
    print("LMI prediction succesful.")
