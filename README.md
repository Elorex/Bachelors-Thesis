# RFI Detection and Characterisation in Radio Astronomy Data

Code developed for my 2024 bachelor's thesis, **“Machine Learning-Based Characterisation of RFI Signals in Radio Astronomical Measurement Data”**, at the University of Würzburg.

Radio-frequency interference (RFI) can obscure faint astronomical signals and reduce the quality of radio-telescope observations. This project explores ways to detect and characterise interference in observations from the **TELAMON** monitoring programme at the **Effelsberg 100 m radio telescope**, using both signal-processing heuristics and unsupervised machine learning.

## What's in this repository?

| Directory | Contents |
| --- | --- |
| `SNR_analyzer/` | Python pipeline for preprocessing MBFITS subscans and ranking anomalies using signal-to-noise ratio (SNR); includes a Dockerfile and requirements file. |
| `Laplace_Convolution/` | Python pipeline for edge-based analysis of time–frequency data using Laplacian convolution; includes a Dockerfile and requirements file. |
| `Pytorch_NVIDIA/` | Preprocessing utilities and exploratory Jupyter notebooks for machine-learning approaches to light curves and edge masks. |
| `demos/` | Example notebooks illustrating the SNR and edge-mask approaches. |

## Methods explored

The analysis begins with extraction of intensity data from MBFITS files, bandpass calibration, noise-calibration correction, and (where appropriate) removal of the astronomical source. It then investigates:

- **Anomaly detection:** SNR-based scoring, edge-mask metrics, PCA, Isolation Forest, and One-Class SVM.
- **Characterisation and clustering:** PCA and t-SNE representations, K-means, DBSCAN, LSTM autoencoders for light curves, and convolutional autoencoders for edge masks.

An important finding was that **simple, domain-specific heuristics were more effective at identifying the targeted conspicuous interference patterns** than the tested generic machine-learning anomaly detectors, which often highlighted other forms of variation. The work is exploratory rather than a production-ready RFI flagging tool.

## Running the code

The two standalone analysis directories each provide a `requirements.txt` and `Dockerfile`. For example, to prepare the SNR environment locally:

```bash
cd SNR_analyzer
python -m pip install -r requirements.txt
```

The entry point is `src/main.py`; the corresponding script in `Laplace_Convolution/` serves the edge-based pipeline. **Before running either script**, configure the input-data path in `src/main.py` and the output/log paths in `src/shared.py`. The scripts expect TELAMON MBFITS observations in a specific directory layout, and the original observational dataset is **not included** here. Some notebooks also depend on their original research environment and may need adaptation.

These instructions describe the repository structure; they are not a claim that the project has been tested on a clean installation.

## Thesis and context

The thesis was submitted in October 2024 as part of the B.Sc. in Aerospace Information Technology at Julius-Maximilians-Universität Würzburg, supervised by Prof. Ingo Scholtes and Prof. Matthias Kadler.

For the scientific background, experimental methodology, results and limitations, see the accompanying bachelor's thesis.

## Data and reuse

The observational data originate from the TELAMON project and are not distributed with this repository. Access and reuse of telescope data are subject to the data owners' terms. No open-source license is specified for the code at present; please contact the author regarding reuse until licensing has been clarified.
