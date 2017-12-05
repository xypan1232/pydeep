y_classes = df_y.idxmax(1, skipna=False)

from sklearn.preprocessing import LabelEncoder

# Instantiate the label encoder
le = LabelEncoder()

# Fit the label encoder to our label series
le.fit(list(y_classes))

# Create integer based labels Series
y_integers = le.transform(list(y_classes))

# Create dict of labels : integer representation
labels_and_integers = dict(zip(y_classes, y_integers))

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
sample_weights = compute_sample_weight('balanced', y_integers)

class_weights_dict = dict(zip(le.transform(list(le.classes_)), class_weights))


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

out = model.predict_proba(x_test)
out = np.array(out)

threshold = np.arange(0.1,0.9,0.1)

acc = []
accuracies = []
best_threshold = np.zeros(out.shape[1])
for i in range(out.shape[1]):
    y_prob = np.array(out[:,i])
    for j in threshold:
        y_pred = [1 if prob>=j else 0 for prob in y_prob]
        acc.append( matthews_corrcoef(y_test[:,i],y_pred))
    acc   = np.array(acc)
    index = np.where(acc==acc.max()) 
    accuracies.append(acc.max()) 
    best_threshold[i] = threshold[index[0][0]]
    acc = []

print "best thresholds", best_threshold
y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])

print("-"*40)
print("Matthews Correlation Coefficient")
print("Class wise accuracies")
print(accuracies)

print("other statistics\n")
total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i]==y_pred[i]).sum() == 5])
print("Fully correct output")
print(total_correctly_predicted)
print(total_correctly_predicted/400.)
