#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tools
import os
import shutil
from scipy import ndimage
from simulator import runForwardSolver


#%% run LMI WM is needed for registration
def runLMI(
    registrationReference,
    patientFlair,
    patientT1,
    patientAffine=None,
    registrationMode="WM",
    padding=10,
):
    atlasPath = "./Atlasfiles"

    reg_ref = registrationReference
    applied_padding = 0
    if padding > 0 and patientAffine is not None:
        applied_padding = padding
        reg_ref, patientAffine = tools.pad_image_and_affine(
            registrationReference, patientAffine, padding
        )
        patientFlair, _ = tools.pad_image_and_affine(patientFlair, patientAffine, padding)
        patientT1, _ = tools.pad_image_and_affine(patientT1, patientAffine, padding)

    wmTransformed, transformedTumor, registration = tools.getAtlasSpaceLMI_InputArray(
        reg_ref,
        patientFlair,
        patientT1,
        atlasPath,
        getAlsoWMTrafo=True,
        patientAffine=patientAffine,
    )

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
    predictedTumorPatientSpace = tools.convertTumorToPatientSpace(
        tumor,
        reg_ref,
        registration,
        patientAffine=patientAffine,
        padding=applied_padding,
    )
    referenceBackTransformed = tools.convertTumorToPatientSpace(
        wmTransformed,
        reg_ref,
        registration,
        patientAffine=patientAffine,
        padding=applied_padding,
    )

    return predictedTumorPatientSpace, parameterDir, referenceBackTransformed

if __name__ == "__main__":

    # Paths
    wmSegmentationNiiPath = "/mlcube_io0/Patient-00000/00000-wm.nii.gz"
    tumorsegPath = "/mlcube_io0/Patient-00000/00000-tumorseg.nii.gz"
    resultPath = "/app/tmp"

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
        patientAffine=patientWMAffine,
    )

    np.save("/mlcube_io1/00000_lmi_parameters.npy", parameterDir)

    nib.save(nib.Nifti1Image(predictedTumorPatientSpace, patientWMAffine), "/mlcube_io1/00000.nii.gz")

    nib.save(nib.Nifti1Image(wmBackTransformed, patientWMAffine), "/mlcube_io1/00000_lmi_wm_patientSpace.nii.gz")
    print("LMI prediction succesful.")
