# import required packages
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



if __name__ == "__main__":

    # Reading the test file
    test_data = pd.read_csv("data/test_data_RNN.csv")

    # Getting labels and features
    X_test = test_data.iloc[:, 0:12]
    y_test = test_data.iloc[:, 13]

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Converting into array was making y_test into undesirable shape. Hence did the reshape.
    y_test = y_test.reshape(-1, 1)

    # Normalizing test data
    trans_test_X = MinMaxScaler(feature_range=(0, 1))
    X_test = trans_test_X.fit_transform(X_test)
    trans_test_y = MinMaxScaler(feature_range=(0, 1))
    y_test = trans_test_y.fit_transform(y_test)

    # Reshaping for the model as LSTM requires a 3D input
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Loading the model
    model = load_model('models/20828608_RNN_model')

    # Getting the prediction on test data. We will get these predictions normalized, hence we do inverse_transform to get the original values.
    predictions = model.predict(X_test)
    final_pred = trans_test_y.inverse_transform(predictions)
    y_test = trans_test_y.inverse_transform(y_test)

    for i in range(len(y_test)):
        y_test[i] = round(y_test[i][0], 2)

    # Getting the Root Mean Squared Loss on the test data
    mse = mean_squared_error(y_test, final_pred)
    rmse = np.sqrt(mse)
    rmse = round(rmse, 3)
    print("Root Mean Squared Error for Test Data: ", rmse)

    # In this section I create a dataframe which I use to plot the actual vs predicted values
    df_values = pd.DataFrame()

    y_test = list(y_test)
    final_predictions = list(final_pred)

    for i in range(len(test_data['date'])):
        test_data['date'][i] = pd.to_datetime(test_data['date'][i]).date()

    # Creating the dataframe which includes Actual value, predicted value and the corresponding date.
    df_values['Actual'] = y_test
    df_values['Predicted'] = final_predictions
    df_values['Date'] = test_data['date']
    df_values = df_values.sort_values(by=['Date'])
    df_values = df_values.reset_index(drop=True)

    # Plotting
    plt.plot(df_values['Date'], df_values['Actual'], color='r', label="Actual Price")
    plt.plot(df_values['Date'], df_values['Predicted'], color='b', label="Predicted Price")
    plt.title('Actual and Predicted Opening Price')
    plt.xlabel('Date')
    plt.ylabel('Opening Price')
    plt.legend()
    plt.show()
