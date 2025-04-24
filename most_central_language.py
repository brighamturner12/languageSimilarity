



import pickle as pkl
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import pandas as pd


def getLanguagesAndDistances( diversityStatus = 3, goodEnoughLength = 5 , numColumns = 12 , numRows = 10 ):
    
    diffs = pd.read_csv("similarities_1.0.tsv",  sep='\t' )
    uniqueLanguages = list(sorted(list(set( 
            list(diffs['LangName_1']) + list(diffs['LangName_2'])
        ))))
    
    european_languages = [
        "Albanian",
        "Lithuanian",
        #"Latvian", 
        "Basque",
        
        "Maltese",
        'Modern Greek',
        
        "Welsh",
        "Irish",
        "Breton",
        "Scottish Gaelic",
        "Hungarian",
        "Finnish",
        #"Estonian",
        
        "German",
        "English",
        "Dutch",
        "Swedish",
        "Danish",
        'Norwegian Bokmål',
        'Norwegian Nynorsk',
        'Western Frisian',
        'Luxembourgish',
        
        'Icelandic',
        'Faroese',
        
        'Italian',
        'French',
        'Spanish',
        'Romanian',
        'Catalan',
        'Portuguese',
        'Galician',
        #'Sardinian',
        'Campidanese Sardinian',
        'Gallurese Sardinian',
        'Logudorese Sardinian',
        'Sassarese Sardinian',
        'Walloon',

        'Occitan',
        'Friulian',
        #picard
        #franco provincial
        #aromanian
        'Asturian',
        'Romansh',
        'Latin',

        'Russian',
        'Polish',
        'Ukrainian',
        'Czech',
        #'Serbian',
        'Bulgarian',
        'Croatian',
        'Slovak',
        'Belarusian',

        #'Bosnian,
        'Slovenian',
        'Macedonian',
        'Silesian',
        #Montenegrin
        'Lower Sorbian',
        'Upper Sorbian',
    ]

    european_languages += ["Esperanto"]
    diffs_european = diffs[ np.logical_and( 
            diffs["LangName_1"].isin(european_languages),
            diffs["LangName_2"].isin(european_languages) 
        ) ]
    
    if diversityStatus == 1:
        diffs_european = diffs_european[
                diffs_european["Robustness"] == "High"
            ]
    elif diversityStatus == 2:
        diffs_european = diffs_european[np.logical_or(
                diffs_european["Robustness"] == "High",
                diffs_european["Robustness"] == "Medium"
            ) ]
    
    resultsz = []
    for lang in european_languages:
        if False:
            lang = "English"
        justThisLang = diffs_european[ 
            np.logical_or( 
            diffs_european["LangName_1"] == lang ,
            diffs_european["LangName_2"] == lang
            ) ] 
        
        if len(justThisLang) >= goodEnoughLength:
            #print( len(justThisLang) )
            justSims = np.array(list(reversed(
                                list(sorted(
                                list(justThisLang['Similarity'])
                                )))))
            basicStep = len(justSims)/(numColumns+1)
            
            thisResult = [lang]
            for i in range(1,numColumns+1):
                idx = int( basicStep * i )
                thisResult.append( justSims[idx] )
            resultsz.append(thisResult)
    
    colNames = ["language"]
    finalNames = []
    for i in range(1,numColumns + 1):
        colNames.append( str(i))
        finalNames.append( str(i) )
    
    dfDistributions = pd.DataFrame( resultsz , columns=colNames )

    allVals = []
    for i in range(numRows):
        theseVals = []
        for j in range( numColumns ):
            theseVals.append(None)
        allVals.append( theseVals )
    finalRankings = pd.DataFrame( allVals ,  columns = finalNames )

    for i in range(1,numColumns+1):
        dfDistributions = dfDistributions.sort_values( str(i) ,ascending=False)

        dfDistributions.index = np.arange( len(dfDistributions) )

        top10Langs = dfDistributions.loc[0:numRows,"language"]

        finalRankings[str(i)] = top10Langs
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)

    print( finalRankings )
            
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

getLanguagesAndDistances( )
