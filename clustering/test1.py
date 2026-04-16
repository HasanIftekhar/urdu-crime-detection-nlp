import codecs
import csv
import pandas
import pandas as pd
indexes=[4, 15, 18, 35, 47, 49, 56, 62, 65, 69]
df1 = pandas.read_csv('SpeechToText.csv')
res = pd.DataFrame()


csv_file =  open("sample_file.csv", 'w',encoding='utf8')
csv_writer = csv.writer(csv_file)
for i in range (0,len(df1)):
    for j in range(len(indexes)):
        if i==indexes[j]:
            text=(df1['0'][i])
            # csv_writer.writerow(text)
            pd.DataFrame(text).to_csv("sample_file.csv")

# df2= pandas.read_csv('sample_file.csv')
# res2= pd.DataFrame()
# print(df2)


