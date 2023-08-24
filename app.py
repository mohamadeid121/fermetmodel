from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import sys

app = Flask(__name__)

if getattr(sys, 'frozen', False):
    # If the script is run as a standalone executable
    bundle_dir = sys._MEIPASS
else:
    # If the script is run in the development environment
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

arima_model_t2 = joblib.load('arima_model_t2.pkl')
arima_model_t3 = joblib.load('arima_model_t3.pkl')
arima_model_t4 = joblib.load('arima_model_t4.pkl')


@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Get the uploaded file from the form
        excel_file_path = os.path.join(bundle_dir, 'over.xlsx')



        # Read the uploaded file into a pandas DataFrame
        df = pd.read_excel(excel_file_path)

        # Get the entered patient ID from the form
        target_patient_id = int(request.form['patient_id'])

        # Get the selected ARIMA option
        arima_option = int(request.form['arima_option'])

        # Initialize a list to store predictions for the target patient
        target_predictions = []

        # Iterate through each row (patient) in the DataFrame
        for index, row in df.iterrows():
            # Determine which ARIMA model to use based on the selected option
            if arima_option == 1:
                arima_model = arima_model_t2
                X_columns = ['Gewicht_T1', 'T2','FFM%_T1','FM%_T1','Pha_T1','RZ_T1','Lengte_T1','Age_T1','SMI_T1','REE_T1','Geslacht_T1']
            elif arima_option == 2:
                arima_model = arima_model_t3
                X_columns = ['Gewicht_T1', 'Gewicht_T2', 'T2', 'T3', 'FFM%_T1', 'FFM%_T2', 'FM%_T1', 'FM%_T2','Pha_T1', 'Pha_T2','RZ_T1', 'RZ_T2', 'Lengte_T1', 'Age_T1', 'SMI_T1', 'REE_T1', 'REE_T2', 'Geslacht_T1']
            elif arima_option == 3:
                arima_model = arima_model_t4
                X_columns =['Gewicht_T1', 'Gewicht_T2', 'Gewicht_T3', 'T2', 'T3', 'T4', 'FFM%_T1', 'FFM%_T2', 'FFM%_T3', 'FM%_T1', 'FM%_T2', 'FM%_T3', 'Pha_T1', 'Pha_T2', 'Pha_T3', 'RZ_T1', 'RZ_T2', 'RZ_T3', 'Lengte_T1', 'Age_T1', 'SMI_T1', 'REE_T1', 'REE_T2', 'REE_T3', 'Geslacht_T1']

            # Prepare the input features (X) for the ARIMA model
            X = row[X_columns]
            X = X.values.reshape(-1, len(X_columns))

            # Get prediction using the selected ARIMA model
            prediction = arima_model.predict(start=len(X), end=len(X), exog=X)

            # If the current row's patient ID matches the target patient ID, add prediction
            if int(row.iloc[0]) == target_patient_id:
                target_predictions.append({
                    'patient_id': row.iloc[0],
                    'prediction': prediction.values[0]
                })

        # Return the list of predictions for the target patient to the user
        return render_template('result.html', predictions=target_predictions, target_patient_id=target_patient_id)

    return render_template('upload.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
