import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(in_file, out_file):
    col_key1 = {"admin.":0,"unknown":1,"unemployed":2,"management":3,"housemaid":4,"entrepreneur":5,"student":6,"blue-collar":7,"self-employed":8,"retired":9,"technician":10,"services":11}
    col_key2 = {"married":12,"divorced":13,"single":14,"unknown":15}
    col_key3 = {"basic.4y":16,"basic.6y":17,"basic.9y":18,"high.school":19,"illiterate":20,"professional.course":21,"university.degree":22,"unknown":23}
    col_key4 = {"yes":24,"no":25,"unknown":26}
    col_key5 = {"yes":27,"no":28,"unknown":29}
    col_key6 = {"yes":30,"no":31,"unknown":32}
    col_key7 = {"telephone":33,"cellular":34}
    col_key8 = {"jan":35,"feb":36,"mar":37,"apr":38,"may":39,"jun":40,"jul":41,"aug":42,"sep":43,"oct":44,"nov":45,"dec":46}
    col_key9 = {"mon":47,"tue":48,"wed":49,"thu":50,"fri":51}
    col_key10 = {"failure":52,"nonexistent":53,"success":54}
    col_key11 = {"yes":1,"no":0}
    
    for l in in_file:
        empty_cols = [0]*55
        line = l.replace('"','').replace('\n','').split(";")
        empty_cols[col_key1[line[1]]] += 1
        empty_cols[col_key2[line[2]]] += 1
        empty_cols[col_key3[line[3]]] += 1
        empty_cols[col_key4[line[4]]] += 1
        empty_cols[col_key5[line[5]]] += 1
        empty_cols[col_key6[line[6]]] += 1
        empty_cols[col_key7[line[7]]] += 1
        empty_cols[col_key8[line[8]]] += 1
        empty_cols[col_key9[line[9]]] += 1
        empty_cols[col_key10[line[14]]] += 1
        newline = str(l[0]) + ","
        for col in empty_cols:
            newline += str(col) + ","
        newline += line[11] + "," + line[12] + "," + line[13] + "," + line[15] + "," + line[16] + "," + line[17] + "," + line[18] + "," + line[19] + "," + str(col_key11[line[20]]) + "\n"
        out_file.write(newline)

def split_norm_data(data_array):
    sorted_data = data_array[data_array[:, -1].argsort()]

    scaler = MinMaxScaler(feature_range=(0,1))
    normalized_data = scaler.fit_transform(sorted_data)

    neg = normalized_data[:36548]
    pos = normalized_data[36548:]

    np.save("normalized_neg_data", neg)
    np.save("normalized_pos_data", pos)

def main():
    # Open data file for read. Open output file for writing.
    in_file = open("bank-additional-full.csv", "r")
    out_file = open("processed_data.csv", "w")
    in_file.readline()
    preprocess_data(in_file, out_file)
    in_file.close()
    out_file.close()

    in_file = open("processed_data.csv", "r")
    in_file.close()
    all_data = np.genfromtxt('processed_data.csv', delimiter=',')
    split_norm_data(all_data)
    pass

if __name__ == "__main__":
    main()