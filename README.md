# CrimeLens — area crime-pattern explorer

A Streamlit prototype for exploring historical incident patterns in the supplied Kerala dataset. Includes a Colab/Kaggle notebook that trains a 1D-CNN crime-category classifier and an LSTM next-hour recorded-incident-count estimator.

## Important limitations
- The supplied data contains recorded incidents, not verified crime-free observations or population/exposure denominators. The models therefore **do not produce calibrated real-world crime probabilities**.
- The dashboard's historical indicator is descriptive, not an official safety rating.
- Dataset place names and coverage determine what can be searched. Unknown names fall back to nearest coordinates when coordinates/live location are provided.
- Do not use the prototype to make law-enforcement decisions, profile people, or guarantee personal safety. Check official local guidance and use emergency services when needed.

## Files
- `app.py` — Streamlit web app
- `data.csv` — provided dataset
- `CrimeLens_Train_CNN_LSTM.ipynb` — training notebook for Google Colab or Kaggle
- `requirements.txt` — dependencies
- `models/` — place generated model files here after training

## Train in Google Colab
1. Upload `CrimeLens_Train_CNN_LSTM.ipynb` to Google Colab.
2. Upload `data.csv` when the notebook asks for it (or add it to the notebook runtime).
3. Run all cells. Training may take several minutes.
4. Download `crimelens_models.zip` from the final cell and unzip it.
5. Upload the files inside the resulting `models/` folder into the `models/` folder in this GitHub repository:
   - `crime_cnn.keras`
   - `crime_lstm.keras`
   - `metadata.json`
   - `hourly_counts.csv`

## Train in Kaggle
1. Create a Kaggle Notebook and add the CSV as an input dataset.
2. Upload/import the `.ipynb`, then change `DATA_PATH` to the Kaggle input path, typically `/kaggle/input/<dataset-folder>/<filename>.csv`.
3. Run all cells. Download `crimelens_models.zip` from the notebook output/files panel.
4. Upload the unzipped model artifacts into GitHub under `models/`.

## GitHub setup
Upload these files/folders to the root of one repository:
- `app.py`
- `requirements.txt`
- `data.csv`
- `CrimeLens_Train_CNN_LSTM.ipynb`
- `models/` (empty before training; add artifacts after training)

Do not upload the dataset to a public repository if you do not have permission to share it.

## Deploy on Streamlit Community Cloud
1. Push/commit the files to GitHub.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**. TensorFlow increases installation time and app memory use; first deploy without trained model files if necessary, then add the artifacts and test.
5. If deployment fails, inspect **Manage app → Logs**. Keep filenames and paths exactly as shown.

## Location permissions
Live location requires HTTPS and browser permission. If the widget does not work in a particular browser, select Coordinates and enter latitude/longitude manually.
