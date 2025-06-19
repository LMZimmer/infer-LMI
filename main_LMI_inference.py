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
    
    # Paths
    wmSegmentationNiiPath = "/mlcube_io0/Patient-00000/00000-wm.nii.gz"
    tumorsegPath = "/mlcube_io0/Patient-00000/00000-tumorseg.nii.gz"
    resultPath = "/app/tmp"

    os.makedirs(resultPath, exist_ok=True)

    # Load tumor core / edema
    tumorAnts = ants.resample_image(
            image=ants.image_read(tumorsegPath),
            resample_params=(129,129,129),
            interp_type=1,
            use_voxels=True
            )
    tumorNib = np.rint(tumorAnts.to_nibabel().get_fdata()).astype(np.uint8)
    patientFlair = (tumorNib==2).astype(np.uint8)
    patientT1 = ((tumorNib==1) | (tumorNib==3)).astype(np.uint8)

    # Load WM, save affine
    patientWMANTS = ants.resample_image(
            image=ants.image_read(wmSegmentationNiiPath),
            resample_params=(129,129,129),
            interp_type=0,
            use_voxels=True
            )
    patientWMNib = patientWMANTS.to_nibabel()
    patientWM = patientWMNib.get_fdata()
    patientWMAffine = patientWMNib.affine

    predictedTumorPatientSpace, parameterDir, wmBackTransformed = runLMI(
            patientWM,
            patientFlair,
            patientT1
            )

    np.save("/mlcube_io1/00000_lmi_parameters.npy", parameterDir)

    nib.save(nib.Nifti1Image(predictedTumorPatientSpace, patientWMAffine), "/mlcube_io1/00000.nii.gz")

    nib.save(nib.Nifti1Image(wmBackTransformed, patientWMAffine), "/mlcube_io1/00000_lmi_wm_patientSpace.nii.gz")
    print("LMI prediction succesful.")
