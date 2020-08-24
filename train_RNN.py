# Import Required Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM


if __name__ == "__main__":
    
    '''
    # Reading the data
    data = pd.read_csv("data/q2_dataset.csv")

    # This is where I create the dataset

    df = pd.DataFrame(
        columns=['vol1', 'open1', 'high1', 'low1', 'vol2', 'open2', 'high2', 'low2', 'vol3', 'open3', 'high3', 'low3'])

    # Using this loop, I start from the earliest date in the q2_dataset and make 3 lists which contain the volume, open, high and low of days.
    # Then I combine these lists and assign to the newly created dataframe above.
    for i in range(len(data) - 1, 2, -1):
        list1 = list(data.iloc[i, [2, 3, 4, 5]].values)
        list2 = list(data.iloc[i - 1, [2, 3, 4, 5]].values)
        list3 = list(data.iloc[i - 2, [2, 3, 4, 5]].values)
        df.loc[i - 3] = list1 + list2 + list3

    df['target(open)'] = np.zeros(1256)
    df['date'] = ' '
    # This is where I append date and target value to the dataframe.
    for i in range(1255, -1, -1):
        df['target(open)'][i] = data.iloc[i, [3]].values
        df['date'][i] = pd.to_datetime(data.iloc[i, [0]].values[0]).date()

    # Randomly shuffle the new dataset
    df = df.sample(frac=1)

    # Train-Test split of dataset using 70-30 split
    train, test = train_test_split(df, test_size=0.3, random_state=42)

    # Creating the required files
    train.to_csv('data/train_data_RNN.csv')
    test.to_csv('data/test_data_RNN.csv')
    '''

    # Reading the train data
    train_data = pd.read_csv("data/train_data_RNN.csv")

    # Getting labels and features
    X_train = train_data.iloc[:, 0:12]
    y_train = train_data.iloc[:, 13]

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Converting into array was making the y_train into undesirable shape. Hence did the reshape.
    y_train = y_train.reshape(-1, 1)

    # Normalization of data to get the values between 0 and 1
    trans_X = MinMaxScaler(feature_range=(0, 1))
    trans_y = MinMaxScaler(feature_range=(0, 1))
    X_train = trans_X.fit_transform(X_train)
    y_train = trans_y.fit_transform(y_train)

    # LSTM requires input in 3D shape. Hence, the reshaping.
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Creating the model
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1, activation = None))
    model.compile(loss='mean_squared_error', optimizer='adam')
   
    # Training the model
    model.fit(X_train, y_train, batch_size=10, epochs=100)

    # Getting the Training loss
    a = model.evaluate(X_train, y_train)
    print("Training Loss: {0:.3g}".format(a))

    # Saving the model
    model.save('models/20828608_RNN_model')
