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
from collections import Counter

#baseFolder = "C:\\Users\\astro\\Desktop\\sentiment work\\Sun Times Reviews\\Roeper\\"
#author = "\\Authors\\Matt Zoller Seitz\\"
baseFolder = "C:\\Users\\astro\\Desktop\\sentiment work\\CS221\\Roger Ebert Reviews\\"
#baseFolder = "C:\\Users\\astro\\Desktop\\sentiment work\\Test\\"
#baseFolder = "C:\\Users\\astro\\Desktop\\sentiment work\\Roger Ebert Reviews\\Authors\\Matt Zoller Seitz\\"
global all_trans_1_dicts
global all_trans_1_totals
global all_trans_2_dicts
global all_trans_2_totals
global all_trans_3_dicts
global all_trans_3_totals
global all_trans_4_dicts
global all_trans_4_totals
global all_files
global sent_pair_dict

global priors
global boundaries
global totalFiles

global splits_equal_prob
global split_simple_occur_prob_prob

score_dict = {}
sent_dict = {}
priors = {}

numSplits = 2
totalScores = 8
scoreIncrements = 0.5
splitSize = totalScores / numSplits
FULL_PRINT = False
window = 1
boundaries = []
prob_comp = 4


#Good to go, just makes boundaries
def make_bound_list():
    global boundaries
    
    if numSplits == 2:
        boundaries = [(0.0, 2.0), (2.5, 4.0)]
    elif numSplits == 3:
        boundaries = [(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)]
    elif numSplits == 4:
        boundaries = [(0.0, 0.5), (1.0, 1.5), (2.0, 2.5), (3.0, 4.0)]
    elif numSplits == 5:
        boundaries = [(0.0, 0.5), (1.0, 1.5), (2.0, 2.5), (3.0, 3.5),
                      (4.0, 4.0)]
    elif numSplits == 6:
        boundaries = [(0.5, 1.0), (1.5, 2.0), (2.5, 2.5), (3.0, 3.0),
                      (3.5, 3.5), (4.0, 4.0)]
    elif numSplits == 7:
        boundaries = [ (0.5, 0.5), (1.0, 1.5), (2.0, 2.0), (2.5, 2.5),
                      (3.0, 3.0), (3.5, 3.5), (4.0, 4.0)]
    elif numSplits == 8:
        boundaries = [(0.5, 0.5), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0),
                      (2.5, 2.5), (3.0, 3.0), (3.5, 3.5), (4.0, 4.0)]

def run_test(wfile_1 = None, wfile_2 = None, wfile_3 = None):
    
    print("Starting Training....")
    make_bound_list()
    sentTraining()
    print("TRAINING DONE\n")

    total_pred = 0
    total_correct = 0
    total_wrong = 0
    off_by = 0.0
    off_by_list = []

    global boundaries
    global totalFiles
    global splits_equal_prob
    global split_simple_occur_prob_prob

    splits_equal_prob = 1 / numSplits
    split_simple_occur_prob = dict.fromkeys(range(numSplits), 0)

    for t in score_dict:
        testScore = score_dict[t]
        #Get current split of test film
        curSplit = -1
        for split in range(len(boundaries)):
            start = (boundaries[split])[0]
            end = (boundaries[split])[1]
            
            if testScore >= start and testScore <= end:
                curSplit = split

        split_simple_occur_prob[curSplit] = split_simple_occur_prob[curSplit] + 1

    for split in split_simple_occur_prob:
        split_simple_occur_prob[split] = (split_simple_occur_prob[split] - 1) / (totalFiles - 1)

    print("Window:",window)
    print("NumSplits:",numSplits)
    print("Boundaries:", boundaries)
    print("Equal Split Prob:", str(splits_equal_prob  * 100))
    print("Simple Split Occur Probabilities:", str(split_simple_occur_prob))
    print()

    splits_equal_prob_count = 0
    score_occur_prob_count = 0

    #print(score_dict)
    for title in score_dict:
        testFilm = title
        testScore = score_dict[testFilm]
        testSent = sent_dict[testFilm]

        #Set up test pairs
        test_pairs = sent_pair_dict[testFilm]
        test_pairs[1] = Counter(test_pairs[1])
        test_pairs[2] = Counter(test_pairs[2])
        test_pairs[3] = Counter(test_pairs[3])
        test_pairs[4] = Counter(test_pairs[4])

        #Get current split of test film
        curSplit = -1
        for split in range(len(boundaries)):
            start = (boundaries[split])[0]
            end = (boundaries[split])[1]
            
            if testScore >= start and testScore <= end:
                curSplit = split

        score_occur_prob = split_simple_occur_prob[curSplit]

        #remove occurences of test film from training
        addRemoveTestPairs(test_pairs, curSplit, isAddBack=False)

        #get test probs
        test_probs = getTestProb(testSent, test_pairs)

        #add back occurences of test film from training
        addRemoveTestPairs(test_pairs, curSplit, isAddBack=True)
                    
        forward, backward, pobs, ppost = calc_fb(test_probs)

        #Ignore all reviews with less than 5 sentences
        if(len(ppost) > 5):
            highest_prob_split = -1
            highest_prob = -1.0
            #All values should be the same (Because not a HMM)
            pp = ppost[0]

            #Get prob of correct split
            corr_split_prob = pp[curSplit]

            #Get sorted list of probs
            prob_list = []
            allZero = True
            for s in pp:
                if(pp[s] > 0.0):
                    allZero = False
                    
                prob_list.append((pp[s], s))

            if allZero:
                prob_list = []
                
                for s in split_simple_occur_prob:
                    prob_list.append((split_simple_occur_prob[s], s))
                    
            prob_list.sort(reverse=True)

            #Pred window by split
            split_list = []
            for i in prob_list:
                split_list.append(i[1])

            highest_prob_split = (prob_list[0])[1]

            #HYPOTHESIS TEST
            if(corr_split_prob > splits_equal_prob):
                splits_equal_prob_count = splits_equal_prob_count + 1
            if(corr_split_prob > score_occur_prob):
                score_occur_prob_count = score_occur_prob_count + 1

            if(FULL_PRINT):
                print("Movie:", testFilm)
                print("Score:", str(testScore))
                print("CurSplit:", curSplit)
                print("Probs:", prob_list)

            #'Do' Prediction
            total_pred = total_pred + 1
            if curSplit in split_list[0:window]:
                total_correct = total_correct + 1
                
                if(FULL_PRINT):
                    print("CORRECT")


            else:
                total_wrong = total_wrong + 1

                diff = (prob_list[0][0] - prob_list[split_list.index(curSplit)][0])
                off_by = off_by + diff
                off_by_list.append(diff)

                if(FULL_PRINT):
                    print("WRONG")
                    print("Off by:", split_list.index(curSplit))


            if(FULL_PRINT):
                print("~~~~~~~~~~~~~~~~")

    off_by_list.sort()
    if( len(off_by_list) > 0):
        median = off_by_list[len(off_by_list) // 2]
    else:
        median = 0.0
                
    print("Total Predictions:", total_pred)
    print("\tTotal Correct:", total_correct)
    print("\tTotal Incorrect:", total_wrong)
    print("\tPercent Correct:", (total_correct / total_pred) * 100, "%")
    
    print("Off By AVG:", (off_by / total_pred) * 100, "%")
    print("Off By MEDIAN:", median * 100, "%")

    print("splits_equal_prob_count:",
          (splits_equal_prob_count / total_pred) * 100, "%")
    print("score_occur_prob_count:",
          (score_occur_prob_count / total_pred) * 100, "%")

    if(wfile_1 != None):
        wfile_1.write(str((total_correct / total_pred) * 100) + ",")
    if(wfile_2 != None):
        wfile_2.write(str((splits_equal_prob_count / total_pred) * 100) + ",")
    if(wfile_3 != None):
        wfile_3.write(str((score_occur_prob_count / total_pred) * 100) + ",")

    print("\n",all_files, "\n")

    #print(all_trans_3_dicts)
    #print("Files1:", files1)
    #print("Files2:", files2)
    #print("TotalFiles:", totalFiles)

def addRemoveTestPairs(test_pairs, curSplit, isAddBack):
    global all_trans_1_dicts
    global all_trans_1_totals
    global all_trans_2_dicts
    global all_trans_2_totals
    global all_trans_3_dicts
    global all_trans_3_totals
    global all_trans_4_dicts
    global all_trans_4_totals
    global all_files
    global totalFiles

    #remove unigram occurences for test film
    for i in test_pairs[1]:
        i_val = (test_pairs[1])[i]

        if(isAddBack):
            (all_trans_1_dicts[curSplit])[int(i)] = (all_trans_1_dicts[curSplit])[int(i)] + i_val
            all_trans_1_totals[curSplit] = all_trans_1_totals[curSplit] + i_val
        else:
            (all_trans_1_dicts[curSplit])[int(i)] = (all_trans_1_dicts[curSplit])[int(i)] - i_val
            all_trans_1_totals[curSplit] = all_trans_1_totals[curSplit] - i_val

    #remove bigram occur for test film
    for i in test_pairs[2]:
        i_val = (test_pairs[2])[i]
        last_gram = i[0]

        if(isAddBack):
            (all_trans_2_dicts[curSplit])[i] = (all_trans_2_dicts[curSplit])[i] + i_val
            ((all_trans_2_totals[curSplit])[last_gram]) = ((all_trans_2_totals[curSplit])[last_gram]) + i_val
        else:
            (all_trans_2_dicts[curSplit])[i] = (all_trans_2_dicts[curSplit])[i] - i_val

            try:
                ((all_trans_2_totals[curSplit])[last_gram]) = ((all_trans_2_totals[curSplit])[last_gram]) - i_val
            except:
                print(i)
                print(last_gram)
                exit()

    #remove 3gram occur for test film
    for i in test_pairs[3]:
        i_val = (test_pairs[3])[i]
        last_gram = i[:-1]

        if(isAddBack):
            (all_trans_3_dicts[curSplit])[i] = (all_trans_3_dicts[curSplit])[i] + i_val
            ((all_trans_3_totals[curSplit])[last_gram]) = ((all_trans_3_totals[curSplit])[last_gram]) + i_val
        else:
            (all_trans_3_dicts[curSplit])[i] = (all_trans_3_dicts[curSplit])[i] - i_val
            ((all_trans_3_totals[curSplit])[last_gram]) = ((all_trans_3_totals[curSplit])[last_gram]) - i_val

    #remove 4gram occur for test film
    for i in test_pairs[4]:
        i_val = (test_pairs[4])[i]
        last_gram = i[:-1]

        if(isAddBack):
            (all_trans_4_dicts[curSplit])[i] = (all_trans_4_dicts[curSplit])[i] + i_val
            ((all_trans_4_totals[curSplit])[last_gram]) = ((all_trans_4_totals[curSplit])[last_gram]) + i_val
        else:
            (all_trans_4_dicts[curSplit])[i] = (all_trans_4_dicts[curSplit])[i] - i_val
            ((all_trans_4_totals[curSplit])[last_gram]) = ((all_trans_4_totals[curSplit])[last_gram]) - i_val

    #decrement total files by one (test file)
    if(isAddBack):
        all_files[curSplit] = all_files[curSplit] + 1
        totalFiles = totalFiles + 1
    else:
        all_files[curSplit] = all_files[curSplit] - 1
        totalFiles = totalFiles - 1


def getTestProb(testSent, test_pairs):
    test_probs = []
    #print(test_pairs)

    #iterate through all sentiments
    for x in range(len(testSent)):
        s = testSent[x]

        if s != "":
            sVal = int(s)

            #If at the first sent, just get overall sent probability of that s
            if x == 0:
                t = {}
                for curSplit in range(numSplits):
                    try:
                        t[curSplit] = ((all_trans_1_dicts[curSplit])[sVal]) / (all_trans_1_totals[curSplit])
                    except:
                        t[curSplit] = 0.0
                    #print(t[curSplit])
                                              
                test_probs.append(t)
                #print(test_probs)
            else:
                #Get bigram
                trans_2 = (int(testSent[x-1]), sVal)

                #Get trigram
                if x - 2 >= 0:
                    trans_3 = (int(testSent[x-2]), int(testSent[x-1]), sVal)
                else:
                    trans_3 = -1

                #get 4gram
                if x - 3 >= 0:
                    trans_4 = (int(testSent[x-3]), int(testSent[x-2]),
                               int(testSent[x-1]), sVal)
                else:
                    trans_4 = -1
                

                t = {}
                for curSplit in range(numSplits):
                    t[curSplit] = -1.0

                    val_array = []

                    #prepare value for one
                    try:
                        one_val = (all_trans_1_dicts[curSplit])[sVal] / all_trans_1_totals[curSplit]
                    except:
                        one_val = 0.0

                    val_array.append(one_val)

                    #prepare value for bigram
                    try:
                        two_val = ((all_trans_2_dicts[curSplit])[trans_2]) / ((all_trans_2_totals[curSplit])[sVal])    
                    except:
                        two_val = 0.0

                    val_array.append(two_val)

                    #prepare value for trigram
                    if trans_3 == -1:
                            three_val = two_val
                    else:
                            try:
                                three_val = ((all_trans_3_dicts[curSplit])[trans_3]) / ((all_trans_3_totals[curSplit])[trans_2])
                            except:
                                three_val = 0.0

                    val_array.append(three_val)

                    #prepare value for 4gram
                    if trans_4 == -1:
                            if trans_3 != -1:
                                four_val = three_val
                            else:
                                four_val = two_val
                    else:
                        try:
                            four_val = ((all_trans_4_dicts[curSplit])[trans_4]) / ((all_trans_4_totals[curSplit])[trans_3])
                        except:
                            four_val = 0.0
                            
                    val_array.append(four_val)

                    #print("\n", one_val, two_val, three_val, four_val)
                    if(one_val < 0):
                        print("\t", sVal, trans_2, trans_3, trans_4)
                        print("\tALL DICT\n\t", all_trans_1_dicts[0])
                        print("\t", all_trans_1_dicts[1])
                        print("\tTOTALS", all_trans_1_totals)

                    #choose optimal prob value based on settings
                    ret_val = -1.0
                    if prob_comp != -1:
                        for x in range(prob_comp):   
                            if val_array[x] > 0.0:
                                ret_val = val_array[x]
                    else:
                        ret_val =  one_val * two_val * three_val * four_val

                    t[curSplit] = ret_val
                                 
                #t = (trans_dict2[trans], trans_dict1[trans])
                test_probs.append(t)

    return test_probs

def calc_fb(test_probs): # don't change this line
    '''test_probs is list of (p(roll|F), p(roll|L)); ptrans is p[from][to];
    prior is (p(F), p(L))'''
    forward = []
    backward = []
    ppost = []
    pobs = []
    test_probs_len = len(test_probs)

    #print(test_pairs)

    global priors
    prior = priors

    #Calculate Forward Probabilities
    for pos_num in range(test_probs_len):
        sent_probs = test_probs[pos_num]
        forward_probs = dict.fromkeys(range(numSplits), 0.0)
        
        if(pos_num == 0):
            for curSplit in range(numSplits):
                forward_probs[curSplit] = prior[curSplit] * sent_probs[curSplit]
        else:
            prev_f = forward[pos_num-1]

            for curSplit in range(numSplits):
                forward_probs[curSplit] = prev_f[curSplit] * sent_probs[curSplit]
                
        forward.append(forward_probs)

    #Calculate Backward Probabilities
    for pos_num in reversed(range(test_probs_len)):
        sent_probs = test_probs[pos_num]
        backward_probs = dict.fromkeys(range(numSplits), 0.0)
        
        if(pos_num == (test_probs_len - 1)):
            for curSplit in range(numSplits):
                backward_probs[curSplit] = 1.0
        else:
            next_b = backward[0]
            next_sent = test_probs[pos_num + 1]

            for curSplit in range(numSplits):
                backward_probs[curSplit] = next_b[curSplit] * next_sent[curSplit]
        
        backward.insert(0,backward_probs)

    #Calculate Posteriors
    for pos_num in range(test_probs_len):
        f = forward[pos_num]
        b = backward[pos_num]
        p_obs = 0.0

        p_posts = {}

        for curSplit in range(numSplits):
            #print("F:",f[curSplit] , "B:", b[curSplit])
            p_obs = p_obs + (f[curSplit] * b[curSplit])

        #print("\t",p_obs)
        for curSplit in range(numSplits):
            #print("\t\t",f[curSplit])
            if p_obs != 0:
                p_posts[curSplit] = (f[curSplit] * b[curSplit]) / p_obs
            else:
                p_posts[curSplit] = 0
        
        pobs.append(p_obs)
        ppost.append(p_posts)

        #print(pobs)
        #print(ppost)
    
    return forward, backward, pobs, ppost



#trains model by getting info from each file
def sentTraining():
    rfile = open(baseFolder + "sentiment_time_comp.csv", "r")
    lines = rfile.readlines()
    print("NUM LINES:", len(lines))
    keys = [-2, -1, 0, 1, 2]
    trans_keys = [(-2,-2), (-2, -1), (-2, 0), (-2,1), (-2,2),
                  (-1,-2), (-1, -1), (-1, 0), (-1,1), (-1,2),
                  (0,-2), (0, -1), (0, 0), (0,1), (0,2),
                  (1,-2), (1, -1), (1, 0), (1,1), (1,2),
                  (2,-2), (2, -1), (2, 0), (2,1), (2,2)]

    global all_trans_1_dicts
    all_trans_1_dicts = {}
    global all_trans_1_totals
    all_trans_1_totals = {}
    
    global all_trans_2_dicts
    all_trans_2_dicts = {}
    global all_trans_2_totals
    all_trans_2_totals = {}
    
    global all_trans_3_dicts
    all_trans_3_dicts = {}
    global all_trans_3_totals
    all_trans_3_totals = {}

    global all_trans_4_dicts
    all_trans_4_dicts = {}
    global all_trans_4_totals
    all_trans_4_totals = {}
    
    global all_files
    all_files = {}

    global sent_pair_dict
    sent_pair_dict = {}

    #initalize all dictionaries
    for split in range(numSplits):
        all_trans_1_dicts[split] = dict.fromkeys(keys, 0)
        all_trans_1_totals[split] = 0
        all_trans_2_dicts[split] = dict.fromkeys(trans_keys, 0)
        all_trans_2_totals[split] = {}
        all_trans_3_dicts[split] = {}
        all_trans_3_totals[split] = {}
        all_trans_4_dicts[split] = {}
        all_trans_4_totals[split] = {}
        all_files[split] = 0

    global totalFiles
    totalFiles = 0

    #iterate through each film
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

        #determine what is the current split for the given film score
        curSplit = -1
        for split in range(len(boundaries)):
            start = (boundaries[split])[0]
            end = (boundaries[split])[1]

            if score >= start and score <= end:
                curSplit = split

        #Increment total num files for given split
        try: 
            all_files[curSplit] = all_files[curSplit] + 1
        except KeyError:
            print(boundaries)
            print(score)
            print(info)

        #Strip any possibleble blank strings from list
        sentiment = [i for i in sentiment if i != '']

        #prep dict to hold all pairs
        sent_pair_dict[title] = {}

        #add unigram to dict
        (sent_pair_dict[title])[1] = sentiment
        
        #Actually iterate through sentiment sequence
        sentTotal = 0

        #Iterate through each sentiment value
        for sNum in range(len(sentiment)):
            s = sentiment[sNum]
            if len(s) > 0:
                sentTotal = sentTotal + 1
                sVal = int(s)

                #Add unigrams
                all_trans_1_totals[curSplit] = all_trans_1_totals[curSplit] + 1
                (all_trans_1_dicts[curSplit])[sVal] = (all_trans_1_dicts[curSplit])[sVal] + 1

                #Add bigrams
                if sNum < (len(sentiment) - 1) and sentiment[sNum + 1] != "":
                    t = (sVal, int(sentiment[sNum + 1]))
                    #add gram to dict
                    try:
                        (sent_pair_dict[title])[2] = ((sent_pair_dict[title])[2]) + [t]
                    except KeyError:
                        (sent_pair_dict[title])[2] = [t]
                    
                    (all_trans_2_dicts[curSplit])[t] = (all_trans_2_dicts[curSplit])[t] + 1

                #Add trigrams
                if sNum < (len(sentiment) - 2) and sentiment[sNum + 2] != "":
                    t = (sVal, int(sentiment[sNum + 1]), int(sentiment[sNum + 2]))
                    #add gram to dict
                    try:
                        (sent_pair_dict[title])[3] = ((sent_pair_dict[title])[3]) + [t]
                    except KeyError:
                        (sent_pair_dict[title])[3] = [t]

                    try:
                        (all_trans_3_dicts[curSplit])[t]
                        (all_trans_3_dicts[curSplit])[t] = (all_trans_3_dicts[curSplit])[t] + 1
                    except KeyError:
                        (all_trans_3_dicts[curSplit])[t] = 1

                #Add 4grams
                if sNum < (len(sentiment) - 3) and sentiment[sNum + 3] != "":
                    t = (sVal, int(sentiment[sNum + 1]),
                         int(sentiment[sNum + 2]), int(sentiment[sNum + 3]))
                    #add gram to dict
                    try:
                        (sent_pair_dict[title])[4] = ((sent_pair_dict[title])[4]) + [t]
                    except KeyError:
                        (sent_pair_dict[title])[4] = [t]

                    try:
                        (all_trans_4_dicts[curSplit])[t]
                        (all_trans_4_dicts[curSplit])[t] = (all_trans_4_dicts[curSplit])[t] + 1
                    except KeyError:
                        (all_trans_4_dicts[curSplit])[t] = 1
                            
                #END CALCS FOR ALL SPLITS    

        #print(title, "::::", sentTotal)
    print("TOTAL FILES: ", totalFiles)

    global priors
    #Get all totals
    for curSplit in range(numSplits):

        for k in range(-2,3):
            trans_2_total = 0
            trans_3_total = 0
            trans_4_total = 0
            
            for i in range(-2,3):
                #increment 2-gram totals
                try:
                    v = (all_trans_2_dicts[curSplit])[(k,i)]
                except KeyError:
                    v = 0

                try:
                    (all_trans_2_totals[curSplit])[(k)] = (all_trans_2_totals[curSplit])[(k)] + v
                except KeyError:
                    (all_trans_2_totals[curSplit])[(k)] = v

                for j in range(-2,3):
                    #increment 3-gram totals
                    try:
                        v = (all_trans_3_dicts[curSplit])[(k,i,j)]
                    except KeyError:
                        v = 0

                    try:
                        (all_trans_3_totals[curSplit])[(k,i)] = (all_trans_3_totals[curSplit])[(k,i)] + v
                    except KeyError:
                        (all_trans_3_totals[curSplit])[(k,i)] = v

                    for h in range(-2,3):
                        #increment 4gram totals
                        try:
                            v = (all_trans_4_dicts[curSplit])[(k,i,j,h)]
                        except KeyError:
                            v = 0

                        try:
                            (all_trans_4_totals[curSplit])[(k,i,j)] = (all_trans_4_totals[curSplit])[(k,i,j)] + v
                        except KeyError:
                            (all_trans_4_totals[curSplit])[(k,i,j)] = v

        #Update prior for the given split
        priors[curSplit] = all_files[curSplit] / totalFiles


def full_test_run():
    global numSplits
    global totalScores
    global splitSize
    global window
    global boundaries
    global prob_comp

    n_grams = [-1, 1, 2, 3, 4]

    for gram in n_grams:
        prob_comp = gram

        resultsFolder = baseFolder + "Results\\"
        if not os.path.exists(resultsFolder):
            os.makedirs(resultsFolder)

        hyp_folder = resultsFolder + "H Tests\\"
        if not os.path.exists(hyp_folder):
            os.makedirs(hyp_folder)          
        
        resultsName1 = resultsFolder + "results_" + str(prob_comp) + "gram.csv"
        resultsName2 = hyp_folder + "h_results_" + str(prob_comp) + "_equal_prob.csv"
        resultsName3 = hyp_folder + "h_results_" + str(prob_comp) + "_occur_prob.csv"
        
        wfile_1 = open(resultsName1, "w")
        wfile_2 = open(resultsName2, "w")
        wfile_3 = open(resultsName3, "w")

        wfile_1.write(",Win1,Win2,Win3,Win4,Win5,Win6,Win7,Win8,Win9\n")
        wfile_2.write(",Win1,Win2,Win3,Win4,Win5,Win6,Win7,Win8,Win9\n")
        wfile_3.write(",Win1,Win2,Win3,Win4,Win5,Win6,Win7,Win8,Win9\n")

        for numS in range(1, 9):
            numSplits = numS
            splitSize = totalScores / numSplits
            boundaries = []
            wfile_1.write("Split " + str(numSplits) + ",")
            wfile_2.write("Split " + str(numSplits) + ",")
            wfile_3.write("Split " + str(numSplits) + ",")
            for w in  range(1,numSplits):
                window = w
                run_test(wfile_1, wfile_2, wfile_3)

            for x in range(10 - numSplits):
                wfile_1.write(str(100) + ",")
                
            wfile_1.write("\n")
            wfile_2.write("\n")
            wfile_3.write("\n")

            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        wfile_1.close()
        wfile_2.close()
        wfile_3.close()


full_test_run()
            
