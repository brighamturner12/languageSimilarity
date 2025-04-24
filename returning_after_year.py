

import pandas as pd
import numpy as np



diffs = pd.read_csv("similarities_1.0.tsv",  sep='\t' )
uniqueLanguages = list(sorted(list(set( list(diffs['LangName_1']) + list(diffs['LangName_2'])))))


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


diffs_european = diffs[ np.logical_and( 
        diffs["LangName_1"].isin(european_languages),
        diffs["LangName_2"].isin(european_languages) 
    ) ]
diffs_european.index = np.arange(len(diffs_european))


toEnglish = diffs_european[np.logical_or(
        diffs_european["LangName_1"] == "English" ,
        diffs_european["LangName_2"] == "English"
        )]

toEnglish = toEnglish.sort_values(by="Similarity",ascending=False)


def selectSimilarity(lang1,lang2):
    assert lang1 in european_languages 
    assert lang2 in european_languages 
    bothh = [lang1,lang2]
    return diffs_european[np.logical_and(
        diffs_european["LangName_1"].isin(bothh) ,
        diffs_european["LangName_2"].isin(bothh)
        )]

print(selectSimilarity("English","Portuguese"))
print(selectSimilarity("Spanish","Portuguese"))
print(selectSimilarity("German","Danish"))
print(selectSimilarity("English","Danish"))


