import sys
#sys.path.insert(0, 'C:\\Users\\astro_000\\git\\RESEARCH_WordModel\\BNC')
#import bncWork

import nltk
import re
import os
import hashlib
import time
import operator
import os.path
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from nltk.cluster.api import ClusterI
from random import randint
from shutil import copyfile


sentDir = "C:\\Users\\astro\\Desktop\\sentiment work\\Roger Ebert Reviews\\Sentiment Texts\\"
baseDir = sentDir + "..\\Authors\\"

def move_files():
    for subdir, dirs, files in os.walk(sentDir):
        for file in files:
            print(file)
            author = str(file).split("_")[1]

            directory = baseDir + author + "\\Sentiment Texts\\"
            
            if not os.path.exists(directory):
                os.makedirs(directory)

            copyfile(sentDir + file, directory + file)

def check_sizes():
    for subdir, dirs, files in os.walk(sentDir):
        for d in dirs:
            print(str(d))
