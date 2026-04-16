from googletrans import Translator
from glob import glob
import speech_recognition as sr
from tinytag import TinyTag
import re
import csv
import codecs


trans = Translator()
data_dir = './New folder/'
audio_files = glob(data_dir + './*.wav')

r = sr.Recognizer()


def audio_metadata_function(audio_files, file):
    tag = TinyTag.get(audio_files[file])
    print("\t\t\t\t-------------AUDIO META DATA-------------")
    print('File size is %.2f Bytes' % tag.filesize)
    print('Bitrate of track is %.2f.' % tag.bitrate)
    print('It is %.2f seconds long.' % tag.duration)

    return tag.duration


def English_to_urdu_translation_txt(text):
    trans = Translator()
    print("\t\t\t-------------ENGLISH TO URDU TRANSLATION-------------")
    print(trans.translate(text, src='en', dest='urdu').text)


def Speech_to_text(audio):
    text = r.recognize_google(audio,language='ur-PK')
    print("\t\t\t\t-------------SPEECH TO TEXT-------------")
    print(text)
    return text

def compraing_Criminal_words(text):
    Criminal_words = ['اثاثوں', ' کیسا', 'قومیں','ٹریجک','دولت','نیشنل','منظر']
    input_sentence = re.findall(r'[\u0600-\u06ff]+', text)
    for i in range(len(Criminal_words)):
        for sentene_wrod in input_sentence:
            if Criminal_words[i] == sentene_wrod:
                print(Criminal_words[i] + ":Word found")


def Storing_translated_text_to_file(text,x):

    file_object = codecs.open('textfile.txt', "a", "utf-8")
    str1 = text.split('\n')

    for n in str1:
        file_object.write(n + "("+ str(x) + u"\n" )

    return "Text Stored in File"

filesize = 0
avg = 0
for file in range(0, len(audio_files), 1):

    with sr.AudioFile(audio_files[file]) as source:

        audio = r.listen(source)
        try:

            print("File Number", file + 1)
            # print("Aduio file name", audio_files[file])
            filesize = filesize + audio_metadata_function(audio_files, file)
            r.adjust_for_ambient_noise(source)
            text = Speech_to_text(audio)
            x=0
            result = Storing_translated_text_to_file(text,x)
            print(result)
            compraing_Criminal_words(text)
            #English_to_urdu_translation_txt(text)

        except:
            print("sorry")

avg = filesize / (file + 1)
print("\t\t\t\t----------------Average-----------------")
print("Avergae Length of Audio Files %.2f " % avg)









