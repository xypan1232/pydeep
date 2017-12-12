import os
import sys

"""
12_PARCLIP_EWSR1_hg19    EWSR1
13_PARCLIP_FUS_hg19    FUS
15_PARCLIP_IGF2BP123_hg19    IGF2BP123
17_ICLIP_HNRNPC_hg19    HNRNPC
18_ICLIP_hnRNPL_Hela_group_3975_all-hnRNPL-Hela-hg19_sum_G_hg19--ensembl59_from_2337-2339-741_bedGraph-cDNA-hits-in-genome    hnRNPL
1_PARCLIP_AGO1234_hg19    AGO1234
21_PARCLIP_MOV10_Sievers_hg19    MOV10
22_ICLIP_NSUN2_293_group_4007_all-NSUN2-293-hg19_sum_G_hg19--ensembl59_from_3137-3202_bedGraph-cDNA-hits-in-genome    NSUN2
23_PARCLIP_PUM2_hg19    PUM2
24_PARCLIP_QKI_hg19    QKI
25_CLIPSEQ_SFRS1_hg19    SFRS1
26_PARCLIP_TAF15_hg19    TAF15
27_ICLIP_TDP43_hg19    TDP43
28_ICLIP_TIA1_hg19    TIA1
29_ICLIP_TIAL1_hg19    TIAL1
30_ICLIP_U2AF65_Hela_iCLIP_ctrl_all_clusters    U2AF65
4_HITSCLIP_Ago2_binding_clusters_2    Ago2
7_CLIP-seq-eIF4AIII_2    eIF4AIII
8_PARCLIP_ELAVL1_hg19    ELAVL1
"""

def get_rbps_name(rbp_file = 'RBPs'):
    rbps = {}
    with open(rbp_file) as fp:
        for line in fp:
            values = line.rstrip().split()
            rbps[values[0]] = values[1]
    return rbps

def read_fasta_file(path_dir):
    fasta_file = path_dir +'/sequences.fa.gz'
    seq_dict = {}    
    fp = gzip.open(fasta_file, 'r')
    name = ''
    name_list = []
    for line in fp:
        line = line.rstrip()
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[2:] #discarding the initial >#chr1,+,69224,69324; class:0
            name_str = name.split(';')
            label = name_str[1].split(':')[-1]
            coors = name_str[0].split(',')
            key = coors[0] + '_' + coors[2] + '_' + coors[3] #+ '_' + label
            name_list.append(name)
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper().replace('U', 'T')
    fp.close()
    
    return seq_dict, name_list

def read_bed_file(path_dir):
    bed_file = path_dir + 'positions.bedGraph.gz'
    bed_posi = {}
    fp = gzip.open(bed_file, 'r')
    for line in fp:
        values = line.rstrip().split()
        start = int(values[1]) - 50
        end = int(values[1]) + 49
        key = values[0] + '_' + str(start) + '_' + str(end) #+ '_' + values[-1]
        bed_posi[key] = int(values[-1])
    
    fp.close()
    
    return bed_posi

def run_predict():
    rbps = get_rbps_name()
    data_dir = '/home/panxy/eclipse/iONMF/datasets/clip/'
    all_bed = {}
    all_seq_dir = {}
    
    for protein, name in iteritems(rbps):
        print protein, name
        path_dir = data_dir + protein + '/30000/training_sample_0'
        all_bed[name] = read_bed_file(path_dir)
        all_seq_dir[name] = read_fasta_file(data_dir)
    
    all_keys = set()
    posi_keys = set()
    peak_rbp = {}
    for key in all_bed:
        for sub_key, val in all_bed[key].iteritems():
            all_keys.add(sub_key)
            if val == 1:
                posi_keys.add(sub_key)
                peak_rbp.setdefault(sub_key, []).append(key)
    all_rbps = rbps.values()
    
    fw = open('binding_seq.fa')
    for site in posi_keys:
        mul_pros = peak_rbp[site]
        labels = ['0'] * len(all_rbps)
        for pro in mul_pros:
            ind = all_rbps.index(pro)
            labels[index] = '1'
        rbp_seqs = all_seq_dir[mul_pros[0]]
        seq = rbp_seqs[site]
        mylabel = "_".join(labels)
        fw.write('>' + site + ',' + mylabel + '\n')
        fw.write(seq + '\n')
    fw.close()
        
        





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
