# Building Ann
ann = Sequential()
ann.add(Dense(units=6, activation="relu"))  # first hidden layer or adding input layer
ann.add(Dense(units=6, activation="relu"))  # adding second hidden layer
ann.add(Dense(units=1, activation="sigmoid"))  # output layer

# traning Ann
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # compiiling
ann.fit(X_train, y_train, batch_size=32, epochs=100)  # training ann on training set
