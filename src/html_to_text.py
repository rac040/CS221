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


def main():
    htmlDir = "C:\\Users\\astro\\Desktop\\sentiment work\\Polygon Reviews\\HTML\\"
    textDir = "C:\\Users\\astro\\Desktop\\sentiment work\\Polygon Reviews\\Text\\"

    for subdir, dirs, files in os.walk(htmlDir):
        for file in files:
            print(file)

            rfile = open(htmlDir + file, "r", encoding="utf8")
            wfile = open(textDir + file.split(".htm")[0] + ".txt", "w")
            lines = rfile.readlines()

            inReview = False
            weWrote = False

            for line in lines:
                if("<div class=\"c-entry-content\">" in line):
                    #print(line)
                    inReview = True

                if("https://www.polygon.com/pages/ethics-statement" in line):
                    #print(line)
                    inReview = False

                if inReview:
                    if("<p " in line):
                        line = re.sub(r'<em>', "", line)
                        line = re.sub(r'</em>', "", line)
                        line = re.sub(r'</p>', "", line)
                        line = re.sub(r'</a>', "", line)
                        lineSplit = line.split(">")
                        
                        for l in lineSplit:
                            if len(l) > 0 and l != "\n":
                                if("<" not in l):
                                    l = re.sub(r'([^A-Za-z\ \'\.\,\"\;\:\?\!\-])', "", l)
                                    l = l.split("\n")[0]
                                    #print(l)
                                    #print("~~~~~~~~~")
                                    wfile.write(l + "\n")
                                    weWrote = True
            if weWrote == False:
                print("NO PRINT: " + file)
            wfile.close()
            rfile.close()

def convAllSunHTML():
    htmlDir = "C:\\Users\\astro\\Desktop\\sentiment work\\Sun Times Reviews\\Roeper\\HTML\\"
    textDir = "C:\\Users\\astro\\Desktop\\sentiment work\\Sun Times Reviews\\Roeper\\Text\\"

    for subdir, dirs, files in os.walk(htmlDir):
        for file in files:
            fileExists = os.path.isfile(textDir + str(file))

            if(not fileExists):
                print(file)
                convSunTimesHTML(htmlDir + str(file), textDir)
            else:
                rfile = open(textDir + str(file), "r")
                lines = rfile.readlines()
                if len(lines) <= 5:
                    print(file)
                convSunTimesHTML(htmlDir + str(file), textDir)
                rfile.close()

def convAllEbertHTML():
    htmlDir = "C:\\Users\\astro\\Desktop\\sentiment work\\Roger Ebert Reviews\\HTML\\"
    textDir = "C:\\Users\\astro\\Desktop\\sentiment work\\Roger Ebert Reviews\\Text\\"

    numWithout = 0

    for subdir, dirs, files in os.walk(htmlDir):
        for file in files:
            if ".html" in file:
                print(file)
                convEbertHTML(htmlDir + file, textDir)


def convEbertHTML(htmlName, textDir):
    rfile = open(htmlName, "r", encoding="utf8")
    lines = rfile.readlines()
    inReview = False
    inAd = False
    inScript = False

    reviewScore = -1
    author = ""

    for line in lines:
        if "itemprop=\"ratingValue" in line:
            s = (line.split("itemprop=\"ratingValue\" content=\"")[1]).split("\"></meta")[0]
            reviewScore = float(s)
        if "<span itemprop=\"author\" itemscope" in line:
            author = (line.split("<span itemprop=\"name\">")[1]).split("</span>")[0]

    movieName = (htmlName.split("\\")[-1]).split(".html")[0]
    wfile = open(textDir + movieName + "_" + author +"_" + str(reviewScore) + ".txt", "w")

    text = ""

    for line in lines:
        #print(line + "\n**************************")
        if "reviewBody" in line:
            inReview = True

        if "<script" in line:
            inScript = True

        if "<article class=\"ad\">" in line and line.split("<article class=\"ad\">")[1] != "\n":
            inAd = True

        if inAd and "</article>" in line:
            inAd = False

        if inScript and "</script>" in line:
            inScript = False

        if "<div class=\"whats-hot\">" in line:
            inReview = False

        if inReview and not inAd and not inScript and "</header>" not in line:
            line = line.split("\n")[0]
            #print(line)
            #print("~~~~~~~~~~~~~~~~~~~~~~~")

            #print(line)
            #print(line,"\n~~~~~~~~~~~~~~~~~~~~\n")
            line = re.sub(r'<em>', "", line)
            line = re.sub(r'</em>', "", line)
            line = re.sub(r'</p>', "", line)
            line = re.sub(r'</a>', "", line)
            line = re.sub(r'<i>', "", line)
            line = re.sub(r'</i>', "", line)
            line = re.sub(r'<b>', "", line)
            line = re.sub(r'</b>', "", line)
            line = re.sub(r';', "\'", line)
            line = re.sub(r'—', " ", line)
            line = re.sub(r'</span', "", line)
            line = re.sub(r'</div', "", line)
            #line = re.sub(r'[\t]+', " ", line)
            lineSplit = line.split(">")

            #isking your life to be  isnt
                    
            for l in lineSplit:
                #print(l)
                l = l.split("<")[0]
                if len(l) > 0 and l != "\n" and not re.match('[ ][ ]+', l):
                    #print(l)
                    if("<" not in l):
                        l = re.sub(r'([^A-Za-z\ \'\.\,\"\;\:\?\!\-])', "", l)
                        l = l.split("\n")[0]
                        #print(l)
                        #print("~~~~~~~~~")
                        if("data.comment.user.membership.shortbio" not in l and "Partner site of The Verge" != l):
                            line = re.sub(r' .', ".", line)
                            wfile.write(l + " ")
                            text = text + l + " "
                            #print("\"",l, "\"\n\n@@@@@@@@@@@@@@@@@@@@@@@@\n")
                            weWrote = True
                            
        if "<article class=\"ad\">" in line:
            inAd = True

    if("." not in text[-5:-1]):
        wfile.write(".")
    wfile.close()
    rfile.close()

                                

def convSunTimesHTML(htmlName, textDir): 
    rfile = open(htmlName, "r", encoding="utf8")
    lines = rfile.readlines()
    inReview = False

    reviewScore = -1
    for line in lines:
        bool1 = ("<h3 class=\"p1\">" in line and "★" in line)
        bool2 = ("<h3 class=\"p1\">" in line and "1⁄2" in line)
        bool3 = ("<p>★" in line) or ("<p>1⁄2" in line)
        bool4 = ("<b>★" in line) or ("<b>1⁄2" in line)
        bool5 = ("<p class=\"p1\">" in line and "★" in line)
        bool6 = ("<p class=\"p1\">" in line and "1⁄2" in line)
        bool7 = bool1 or bool2 or bool3 or bool4 or bool5 or bool6
       
        if "★" in line or ("<h3>★" in line) or ("<h3>1⁄2" in line) or bool7:
            reviewScore = line.count("★")
            if "1⁄2" in line:
                reviewScore = reviewScore + 0.5

        checkLine = line.lower()
        bool8 = "<h3>" in line and "zero stars" in checkLine
        bool9 = "<p><strong>" in line and "zero stars" in checkLine

        if bool8 or bool9:
            reviewScore = 0

        if  "s3r star=" in line:
            reviewScore = float(((line.split("s3r star=")[1]).split("/")[0]).split("]")[0])

    movieName = (htmlName.split("\\")[-1]).split(".html")[0]
    movieName = movieName + "_" + str(reviewScore)
    movieName = re.sub(r'\.0', "", movieName)
    movieName = re.sub(r'\.', "-", movieName)

    wfile = open(textDir + movieName + ".txt", "w")
    

    for line in lines:
        if "articleBody" in line:
            inReview = True
        bool1 = ("<h3 class=\"p1\">" in line and "★" in line)
        bool2 = ("<h3 class=\"p1\">" in line and "1⁄2" in line)
        bool3 = ("<p>★" in line) or ("<p>1⁄2" in line)
        bool4 = ("<b>★" in line) or ("<b>1⁄2" in line)
        bool5 = ("<p class=\"p1\">" in line and "★" in line)
        bool6 = ("<p class=\"p1\">" in line and "1⁄2" in line)
        bool7 = bool1 or bool2 or bool3 or bool4 or bool5 or bool6
        checkLine = line.lower()
        bool8 = "<h3>" in line and "zero stars" in checkLine
        bool9 = "<p><strong>" in line and "zero stars" in checkLine
        bool10 = "s3r star=" in line
        if "★" in line or ("<h3>★" in line) or ("<h3>1⁄2" in line) or bool7 or bool8 or bool9 or bool10:
            #print(line)
            inReview = False

        if inReview:
            #print(line)
            if "</p>" in line or "</div>" in line:
                #print(line,"\n~~~~~~~~~~~~~~~~~~~~\n")
                line = re.sub(r'<em>', "", line)
                line = re.sub(r'</em>', "", line)
                line = re.sub(r'</p>', "", line)
                line = re.sub(r'</a>', "", line)
                line = re.sub(r'<i>', "", line)
                line = re.sub(r'</i>', "", line)
                line = re.sub(r'<b>', "", line)
                line = re.sub(r'</b>', "", line)
                line = re.sub(r';', "\'", line)
                line = re.sub(r'</span', "", line)
                line = re.sub(r'</div', "", line)
                lineSplit = line.split(">")
                        
                for l in lineSplit:
                    #print(l)
                    if len(l) > 0 and l != "\n":
                        #print(l)
                        if("<" not in l):
                            l = re.sub(r'([^A-Za-z\ \'\.\,\"\;\:\?\!\-])', "", l)
                            l = l.split("\n")[0]
                            #print(l)
                            #print("~~~~~~~~~")
                            if("data.comment.user.membership.shortbio" not in l and "Partner site of The Verge" != l):
                                wfile.write(l + "\n")
                                #print(l, "\n\n@@@@@@@@@@@@@@@@@@@@@@@@\n")
                                weWrote = True


def convOldFormat():
    htmlDir = "C:\\Users\\astro\\Desktop\\sentiment work\\Polygon Reviews\\Old HTML Format\\"
    textDir = "C:\\Users\\astro\\Desktop\\sentiment work\\Polygon Reviews\\Text\\"

    for subdir, dirs, files in os.walk(htmlDir):
        for file in files:
            print(file)

            rfile = open(htmlDir + file, "r", encoding="utf8")
            wfile = open(textDir + file.split(".htm")[0] + ".txt", "w")
            lines = rfile.readlines()

            inReview = False
            weWrote = False

            for line in lines:
                longString = "<div id=\"div-gpt-ad-tablet_half_page\" class=\"dfp_ad\" "
                longString = longString + "data-cb-dfp-id=\"unit=tablet_half_page\" "
                longString = longString + "data-cb-ad-id=\"Tablet half page\"></div>"
                if("<p class=\"m-entry__intro\">" in line or longString in line):
                    inReview = True
                    if longString in line:
                        line = "<p>" + line.split(longString)[1]
                    if "<p class=\"m-entry__intro\">" in line:
                        line = "<p>" + line.split("<p class=\"m-entry__intro\">")[1]
                    #print(line)

                if("https://www.polygon.com/pages/ethics-statement" in line):
                    #print(line)
                    inReview = False

                if inReview:
                    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    
                    if("<p" in line):
                        #print(line)
                        line = re.sub(r'<em>', "", line)
                        line = re.sub(r'</em>', "", line)
                        line = re.sub(r'</p>', "", line)
                        line = re.sub(r'</a>', "", line)
                        line = re.sub(r'<i>', "", line)
                        line = re.sub(r'</i>', "", line)
                        line = re.sub(r'<b>', "", line)
                        line = re.sub(r'</b>', "", line)
                        lineSplit = line.split(">")
                        
                        for l in lineSplit:
                            #print(l)
                            if len(l) > 0 and l != "\n":
                                if("<" not in l):
                                    l = re.sub(r'([^A-Za-z\ \'\.\,\"\;\:\?\!\-])', "", l)
                                    l = l.split("\n")[0]
                                    #print(l)
                                    #print("~~~~~~~~~")
                                    if("data.comment.user.membership.shortbio" not in l and "Partner site of The Verge" != l):
                                        wfile.write(l + "\n")
                                        weWrote = True
            if weWrote == False:
                print("NO PRINT: " + file)
            wfile.close()
            rfile.close()
