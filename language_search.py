

import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy

#diffs = pd.read_csv("C:/Users/TestAccount/Downloads/similarities_1.0.tsv/similarities_1.0.tsv",  sep='\t' )
diffs = pd.read_csv("similarities_1.0.tsv",  sep='\t' )
uniqueLanguages = list(sorted(list(set( list(diffs['LangName_1']) + list(diffs['LangName_2'])))))


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




