import sys
#sys.path.insert(0, 'C:\\Users\\astro_000\\git\\RESEARCH_WordModel\\BNC')
#import bncWork

import nltk
import re
import os
import hashlib
import time
import operator
import math
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from nltk.cluster.api import ClusterI
from random import randint
from shutil import copyfile

baseFolder = "C:\\Users\\astro\\Desktop\\sentiment work\\Sun Times Reviews\\Roeper\\"
global all_dicts
global all_totals
global all_trans_dicts
global all_trans_totals
global all_files

global priors
global totals_1
global dict1
global trans_dict1
global trans_total1
global totals_2
global dict2
global trans_dict2
global trans_total2
global totalFiles
global files1
global files2

score_dict = {}
sent_dict = {}
priors = []

thresh = 2
numSplits = 2
totalScores = 9
scoreIncrements = 0.5
splitSize = totalScores / numSplits

def run_test():
    
    print("Starting Training....")
    sentAnalysis()
    print("TRAINING DONE\n")

    total_pred = 0
    total_correct = 0
    total_wrong = 0

    #print(score_dict)
    for title in score_dict:
        testFilm = title
        testScore = score_dict[testFilm]
        testSent = sent_dict[testFilm]

        test_probs = getTestProb(testSent)
                    
        forward, backward, pobs, ppost = calc_fb(test_probs, priors)

        if(len(ppost) > 10):
            prob_high = (ppost[0][0] * 100)
            prob_low = (ppost[0][1] * 100)
            #print("Movie:", testFilm)
            #print("Score:", str(testScore))
            #print("Prob of High Article:", prob_high, "%")
            #print("Prob of Low Article:", prob_lox, "%")
            #print()

            total_pred = total_pred + 1
            if testScore <= thresh and (prob_low > prob_high):
                total_correct = total_correct + 1
            elif testScore > thresh and (prob_low > prob_high):
                total_wrong = total_wrong + 1
            elif testScore > thresh and (prob_low < prob_high):
                total_correct = total_correct + 1
            elif testScore <= thresh and (prob_low < prob_high):
                total_wrong = total_wrong + 1
                
    print("Total Predictions:", total_pred)
    print("\tTotal Correct:", total_correct)
    print("\tTotal Incorrect:", total_wrong)
    print("\tPercent Correct:", (total_correct / total_pred) * 100)
    print("\n",all_files)
    print("Files1:", files1)
    print("Files2:", files2)
    print("TotalFiles:", totalFiles)

def getTestProb(testSent):
    test_probs = []
    for x in range(len(testSent)):
        s = testSent[x]

        if s != "":
            sVal = float(s)
            if x == 0:
                t = []
                for curSplit in range(numSplits):
                    t.insert(0, (all_dicts[curSplit])[sVal])
                                              
                #t = (dict2[sVal], dict1[sVal])
                test_probs.append(t)
            else:
                trans = (float(testSent[x-1]), sVal)

                t = []
                for curSplit in range(numSplits):
                    t.insert(0, (all_trans_dicts[curSplit])[trans])
                                 
                #t = (trans_dict2[trans], trans_dict1[trans])
                test_probs.append(t)

    return test_probs

def calc_fb(liks, prior): # don't change this line
    '''liks is list of (p(roll|F), p(roll|L)); ptrans is p[from][to];
    prior is (p(F), p(L))'''
    forward = []
    backward = []
    ppost = [] # replace this with your calculation
    pobs = []
    liks_len = len(liks)
    
    for roll_num in range(liks_len):
        roll = liks[roll_num]
        roll_F = roll[0]
        roll_L = roll[1]
        f_F = 0.0
        f_L = 0.0
        
        if(roll_num == 0):
            f_F = prior[0] * roll_F
            f_L = prior[1] * roll_L
        else:
            prev_f = forward[roll_num-1]
            f_F = (prev_f[0] * 1 * roll_F) + (prev_f[1] * 0 * roll_F)
            f_L = (prev_f[0] * 0 * roll_L) + (prev_f[1] * 1 * roll_L)
                
        forward.append((f_F,f_L))
    
    for roll_num in reversed(range(liks_len)):
        roll = liks[roll_num]
        roll_F = roll[0]
        roll_L = roll[1]
        b_F = 0.0
        b_L = 0.0
        
        if(roll_num == (liks_len - 1)):
            b_F = 1.0
            b_L = 1.0
        else:
            next_b = backward[0]
            next_roll = liks[roll_num + 1]
            b_F = (next_b[0] * 1 * next_roll[0]) + (next_b[1] * 0 * next_roll[1])
            b_L = (next_b[1] * 1 * next_roll[1]) + (next_b[0] * 0 * next_roll[0])
        
        backward.insert(0,(b_F,b_L))
        
    for roll_num in range(liks_len):
        f = forward[roll_num]
        b = backward[roll_num]
        p_obs = (f[0]*b[0]) + (f[1]*b[1])
        
        p_post_f = (f[0]*b[0]) / p_obs
        p_post_b = (f[1]*b[1]) / p_obs
        
        pobs.append(p_obs)
        ppost.append((p_post_f, p_post_b))
    
    return forward, backward, pobs, ppost

def sentAnalysis():
    rfile = open(baseFolder + "sentiment_time_comp.csv", "r", encoding="utf8")
    lines = rfile.readlines()
    keys = [-2, -1, 0, 1, 2]
    trans_keys = [(-2,-2), (-2, -1), (-2, 0), (-2,1), (-2,2),
                  (-1,-2), (-1, -1), (-1, 0), (-1,1), (-1,2),
                  (0,-2), (0, -1), (0, 0), (0,1), (0,2),
                  (1,-2), (1, -1), (1, 0), (1,1), (1,2),
                  (2,-2), (2, -1), (2, 0), (2,1), (2,2)]

    global all_dicts
    all_dicts = {}
    global all_totals
    all_totals = {}
    global all_trans_dicts
    all_trans_dicts = {}
    global all_trans_totals
    all_trans_totals = {}
    global all_files
    all_files = {}

    for split in range(numSplits):
        all_dicts[split] = dict.fromkeys(keys, 0)
        all_totals[split] = 0
        all_trans_dicts[split] = dict.fromkeys(trans_keys, 0)
        all_trans_totals[split] = 0
        all_files[split] = 0

    global totals_1
    totals_1 = 0
    global dict1
    dict1 = dict.fromkeys(keys, 0)
    global trans_dict1
    trans_dict1 = dict.fromkeys(trans_keys, 0)
    global trans_total1
    trans_total1 = 0

    global totals_2
    totals_2 = 0
    global dict2
    dict2 = dict.fromkeys(keys, 0)
    global trans_dict2
    trans_dict2 = dict.fromkeys(trans_keys, 0)
    global trans_total2
    trans_total2 = 0

    global totalFiles
    totalFiles = 0
    global files1
    files1 = 0
    global files2
    files2 = 0

    for line in lines:
        #Get Current movie info
        info = (line.split("\n")[0]).split(",")
        title = info[0]
        try:
            score = float(info[1])
        except ValueError:
            print("ERROR: ",info[1])
            print(info)
            break
        score_dict[title] = score 
        sentiment = info[2:-1]
        sent_dict[title] = sentiment

        #Increment total num file counters
        totalFiles = totalFiles + 1
        if score <= 2:
            files1 = files1 + 1
        else:
            files2 = files2 + 1

        curSplit = -1
        for split in range(numSplits):
            if(split == 0):
                start = 0.0
            else:
               start = (math.ceil(split * splitSize) * scoreIncrements)
            end = (((math.ceil(split * splitSize) + math.ceil(splitSize))) * scoreIncrements) - scoreIncrements

            #print(start, ",", end)
            if score >= start and score <= end:
                curSplit = split
        all_files[curSplit] = all_files[curSplit] + 1


        #Actually iterate through sentiment sequence
        sentTotal = 0
        for sNum in range(len(sentiment)):
            s = sentiment[sNum]
            
            if s != "":
                sentTotal = sentTotal + 1
                sVal = int(s)


                #DO CALCS FOR ALL SPLITS
                all_totals[curSplit] = all_totals[curSplit] + 1
                (all_dicts[curSplit])[sVal] = (all_dicts[curSplit])[sVal] + 1

                if sNum < (len(sentiment) - 1) and sentiment[sNum + 1] != "":
                        t = (sVal, int(sentiment[sNum + 1]))
                        all_trans_totals[curSplit] = all_trans_totals[curSplit] + 1
                        (all_trans_dicts[curSplit])[t] = (all_trans_dicts[curSplit])[t] + 1     
                #END CALCS FOR ALL SPLITS
    
                if score <= thresh:
                    totals_1 = totals_1 + 1
                    dict1[sVal] = dict1[sVal] + 1

                    if sNum < (len(sentiment) - 1) and sentiment[sNum + 1] != "":
                        t = (sVal, int(sentiment[sNum + 1]))
                        trans_dict1[t] = trans_dict1[t] + 1
                        trans_total1 = trans_total1 + 1
                else:
                    totals_2 = totals_2 + 1
                    dict2[sVal] = dict2[sVal] + 1

                    if sNum < (len(sentiment) - 1) and sentiment[sNum + 1] != "":
                        t = (sVal, int(sentiment[sNum + 1]))
                        trans_dict2[t] = trans_dict2[t] + 1
                        trans_total2 = trans_total2 + 1
           

        #print(title, "::::", sentTotal)

    for curSplit in range(numSplits):
        (all_dicts[curSplit])[sVal] = (all_dicts[curSplit])[sVal] / all_totals[curSplit]

        for k in range(-2,3):
            total = 0
            for i in range(-2,3):
                val = (all_trans_dicts[curSplit])[(k,i)]
                total = total + val

            for i in range(-2,3):
                (all_trans_dicts[curSplit])[(k,i)] = (all_trans_dicts[curSplit])[(k,i)] / total

        pri = all_files[curSplit] / totalFiles
        global priors
        priors.insert(0,pri)

    for k in dict1:
        dict1[k] = (dict1[k] / totals_1)

    for k in dict2:
        dict2[k] = (dict2[k] / totals_2)

    for k in range(-2,3):
        total = 0
        for i in range(-2,3):
            val = trans_dict1[(k,i)]
            total = total + val

        for i in range(-2,3):
            trans_dict1[(k,i)] = trans_dict1[(k,i)] / total

    for k in range(-2,3):
        total = 0
        for i in range(-2,3):
            val = trans_dict2[(k,i)]
            total = total + val

        for i in range(-2,3):
            trans_dict2[(k,i)] = trans_dict2[(k,i)] / total

    prior1 = files1 / totalFiles
    prior2 = files2 / totalFiles

    global priors
    priors = (prior1,prior2)
