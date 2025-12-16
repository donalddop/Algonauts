# Algonauts Project (2019) - Bachelor Thesis Code

This repository contains the code written for a bachelor thesis based on the [2019 Algonauts Project](http://algonauts.csail.mit.edu/2019/). The project's goal is to analyze fMRI and MEG data to understand how the brain processes visual information by comparing brain recordings with computational models of vision.

## Disclaimer

This code was written for a bachelor thesis and served as a testbed for different modeling ideas. The code is not well-documented or structured for easy use. It contains many hardcoded paths and commented-out experimental code. The explanations for the code are found within the thesis, which can be found here: [Thesis_DNikkessen_final.pdf](https://staff.fnwi.uva.nl/a.visser/education/bachelorProjects/Thesis_DNikkessen_final.pdf)

## Project Overview

The core idea of this project is to create computational models and compare their internal representations with brain data using Representational Dissimilarity Matrices (RDMs). An RDM is a matrix that represents the pairwise dissimilarity between the brain's responses to a set of stimuli.

This project implements several types of models to generate RDMs:
*   **Perceptual Model:** A simple model based on low-level image features (Canny edge detection).
*   **Categorical Model:** A model based on classifying images into categories using a Gaussian Naive Bayes classifier trained on DNN features.
*   **DNN Model:** A model that directly uses features from a deep neural network (e.g., VGG, AlexNet) to create RDMs.

The generated RDMs are then evaluated by computing their correlation with the target fMRI and MEG RDMs provided by the Algonauts project.

## File Descriptions

*   `model.py`: The main script containing the core logic for loading data, building models, generating RDMs, and running experiments.
*   `testSub_fmri.py`: A script to evaluate model RDMs against fMRI data.
*   `testSub_meg.py`: A script to evaluate model RDMs against MEG data.
*   `Data_Download_TestSets/`: A directory that contains the test datasets from the Algonauts project.
*   `Feature_Extract/`: A directory containing scripts to extract features from deep neural networks.
*   `*.mat`: These files contain data in MATLAB format, including the target RDMs from the fMRI and MEG recordings, and pre-computed DNN features.

## Dependencies

The project was written in Python and relies on the following libraries:

*   PyTorch
*   Torchvision
*   Scikit-learn
*   Scikit-image
*   SciPy
*   NumPy
*   Matplotlib
*   h5py
*   ImageIO

Install the dependencies using pip:
```bash
pip install torch torchvision scikit-learn scikit-image scipy numpy matplotlib h5py imageio
```

## Usage

The main script to run is `model.py`. However, due to the nature of the code, it is not a "plug-and-play" script. The `if __name__ == "__main__":` block in `model.py` contains many different code blocks for running various experiments. To run a specific experiment, you would need to:

1.  **Download the data:** Obtain the necessary data files from the Algonauts 2019 project website and place them in the correct directories as expected by the hardcoded paths in the scripts.
2.  **Modify `model.py`:** Uncomment or modify the desired section of code in the `if __name__ == "__main__":` block in `model.py`.
3.  **Run the script:**
    ```bash
    python model.py
    ```
