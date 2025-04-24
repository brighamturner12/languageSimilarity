





import pickle as pkl
import numpy as np
from scipy.optimize import minimize

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

def getLanguagesAndDistances( diversityStatus = 3, goodEnoughLength = 5 , numColumns = 12 , numRows = 10 ):
    '''
    diversityStatus = 3 ; goodEnoughLength = 5 ; numColumns = 12 ; numRows = 10
    '''

    diffs = pd.read_csv("similarities_1.0.tsv",  sep='\t' )
    
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
    lengthLangs = []
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
            lengthLangs.append( len(justThisLang) )
            justSims = np.array(list(reversed(
                                list(sorted(
                                list(justThisLang['Similarity'])
                                )))))
            
            basicStep = 1 / len( justThisLang )
            
            #thePoinstsz = []
            xsz = []
            ysz = []
            for idx in range( len( justThisLang ) ):
                #thePoinstsz.append( [ idx*basicStep , justSims[idx] ] )
                xsz.append( idx*basicStep )
                ysz.append( justSims[idx] )
            #resultsz.append( [ lang , thePoinstsz ] )
            resultsz.append( [ lang , xsz , ysz ] )

    print( pd.Series(lengthLangs).value_counts() )

    fig = px.line()

    for resultt in resultsz:
        fig.add_trace(go.Scatter(
            x=resultt[1], 
            y=resultt[2], 
            mode='lines', 
            name=resultt[0]))
    
    fig.update_layout(xaxis=dict(
        tickvals=[ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ],
        ticktext=[ '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%' ]))

    #fig.update_layout(legend={'x':1,'y':1.0})
    fig.update_layout(
        #xaxis_range=['2018-05-30','2020-05-31'],
        legend=dict(x=-0.01, y=-0.9),
        legend_orientation="h")
    
    fig.show()

    #######################
    #######################

    import plotly.express as px
    import plotly.graph_objects as go

    x = [1, 2, 3, 4, 5]
    y1 = [10, 14, 18, 24, 30]
    y2 = [5, 8, 12, 15, 20]
    y3 = [2, 6, 9, 12, 15]

    fig = px.line()

    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Series 1'))
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Series 2'))
    fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='Series 3'))

    fig.show()

    #######################
    #######################

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



