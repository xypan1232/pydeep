
clf = Sequential()

clf.add(Dropout(0.3))
clf.add(Dense(xt.shape[1], 1600, activation='relu'))
clf.add(Dropout(0.6))
clf.add(Dense(1600, 1200, activation='relu'))
clf.add(Dropout(0.6))
clf.add(Dense(1200, 800, activation='relu'))
clf.add(Dropout(0.6))
clf.add(Dense(800, yt.shape[1], activation='sigmoid'))

clf.compile(optimizer=Adam(), loss='binary_crossentropy')

clf.fit(xt, yt, batch_size=64, nb_epoch=300, validation_data=(xs, ys), class_weight=W, verbose=0)

preds = clf.predict(xs)

preds[preds>=0.5] = 1
preds[preds<0.5] = 0

print f1_score(ys, preds, average='macro')
