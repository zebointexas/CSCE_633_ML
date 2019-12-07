import math
import numpy as np
import pandas as pd

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols):

        dataframe = pd.read_csv(filename)

        i_split = int(len(dataframe) * split)

        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]

        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)

        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []

        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])  # move windows

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0

       # print("Check point generate_train_batch 1  ******************")
        while i < (self.len_train - seq_len):

            x_batch = []
            y_batch = []

         #   print("Check point generate_train_batch 2 ******************")

            for b in range(batch_size):

             #       print("Check point generate_train_batch 3 ******************")

                    if i >= (self.len_train - seq_len):
                        # stop-condition for a smaller final batch if data doesn't divide evenly
                        yield np.array(x_batch), np.array(y_batch)
                        i = 0

              #      print("Check point generate_train_batch 4 ******************")

                    x, y = self._next_window(i, seq_len, normalise)

              #      print( x.shape)

               #     print("Check point generate_train_batch 5 ******************")

                    x_batch.append(x)
                    y_batch.append(y)

           #         print("Check point generate_train_batch 6 ******************")

                  #  print(x_batch)

                    i += 1

            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''


        window = self.data_train[i:i+seq_len]

      #  print("window 's size = " + str(window.shape))


        window = self.normalise_windows(window, single_window=True)[0] if normalise else window

    #    print("x.shape = " + str(window.shape))

        x = window[:-1]  # 除了最後一個元素的slice

    #    print("x.shape = " + str(x.shape))

        y = window[-1, [0]]

   #     print("y.shape = " + str(y.shape))

  #      print("x 's size = "+ str(x.shape))
   #     print("y 's size = "+ str(y.shape))

        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data

        for window in window_data:

            normalised_window = []

            for col_i in range(window.shape[1]):

                a = float(window[0, col_i])

            #     print("window[0, col_i] = " + str(a))

                if( a == 0):
                    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                    a = 200

                normalised_col = [  (   (   float(p) / a   ) - 1  ) for p in window[:, col_i]]

                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)