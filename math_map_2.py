

import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy

#diffs = pd.read_csv("C:/Users/TestAccount/Downloads/similarities_1.0.tsv/similarities_1.0.tsv",  sep='\t' )
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
# for coll in european_languages:
#     print( coll in uniqueLanguages )

#european_languages = uniqueLanguages 

diffs_european = diffs[ np.logical_and( 
        diffs["LangName_1"].isin(european_languages),
        diffs["LangName_2"].isin(european_languages) 
    ) ]
diffs_european.index = np.arange(len(diffs_european))

matrx = np.zeros( (len(european_languages), len(european_languages) ))

for i in range( len(diffs_european) ):
    indx1 = european_languages.index( diffs_european.loc[ i , "LangName_1" ] )
    indx2 = european_languages.index( diffs_european.loc[ i , "LangName_2" ] )

    matrx[ indx1 , indx2 ] = diffs_european.loc[ i , 'Similarity' ]
    matrx[ indx2 , indx1 ] = diffs_european.loc[ i , 'Similarity' ]

maxx = max(matrx.reshape(-1)) + 1 # + 3
for i in range(len(european_languages)):
    matrx[ i,i ] = maxx 

matrx = matrx / maxx


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform


dissimilarity = 1 - matrx #abs(correlations)

Z_o = linkage(squareform(dissimilarity), 'single')
Z = Z_o.copy()

startt = np.array(pd.Series(Z_o[:, 2]).rank())
# #Z[:, 2] =  np.minimum( max(startt) - 80 , startt )
Z[:, 2] =  startt 

#plt.figure(figsize=(30,100))
plt.figure(figsize=(10,20))

dendrogram(Z, labels=european_languages , orientation='left', 
           leaf_rotation=0 , leaf_font_size=15 ) #was 90

plt.show()


"Georgian" in uniqueLanguages

if False: #pure hate
    ######################

    import seaborn

    matrixDf = pd.DataFrame( matrx , european_languages )

    seaborn.clustermap( matrixDf ,   method="single" , metric="correlation",
                    )
    ####################






    #'Campidanese Sardinian',
    similarity_matrix = -1 * matrx.copy()

    names = european_languages.copy()

    linkage_matrix = hierarchy.linkage(similarity_matrix, method='single')

    # Display the dendrogram
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(20, 12))  # Increase the figure size
    #plt.figure(figsize=(100, 60))  # Increase the figure size

    dendrogram = hierarchy.dendrogram(linkage_matrix, labels=names, orientation='top', distance_sort='descending')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Categories')
    plt.ylabel('Distance')
    plt.show()

    #########################################
    #########################################
    #########################################

    # data =  diffs[['Similarity']]
    # linkage_matrix = linkage(data, method='ward')

    # plt.figure(figsize=(12, 6))
    # dendrogram(linkage_matrix, labels = diffs['LangName_1'] + '-' + diffs['LangName_2'], leaf_rotation=90)

    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Language Pairs')

    # plt.ylabel('Distance')
    # plt.show()





    ##############################


    langName = "Spanish"
    langName = "Hebrew"
    langName = "Lithuanian"

    langName = "Albanian"
    langName = "Portuguese"

    langName = "Chinese"
    langName = 'Sanskrit'
    langName = 'Samoan'
    langName = 'Tuvalu'
    langName = 'Norwegian Bokmål'

    langName = 'English'

    langName = 'Portuguese'
    #diffs["LangName_1"].unique()

    justHebrew = diffs[ np.logical_or(
            diffs["LangName_1"] == langName ,
            diffs["LangName_2"] == langName )]

    justHebrew = justHebrew.sort_values("Similarity",ascending=False)

    justHebrew.index = np.arange( len(justHebrew) )



    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)


    print(  justHebrew  )


    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')




