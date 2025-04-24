

import pickle as pkl
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import pandas as pd

if False:
    def getLanguagesAndDistances():
        
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

        diffs_european = diffs[ np.logical_and( 
                diffs["LangName_1"].isin(european_languages),
                diffs["LangName_2"].isin(european_languages) 
            ) ]
        diffs_european.index = np.arange(len(diffs_european))

        matrx = np.full( (len(european_languages), len(european_languages) ) , -1.0 )

        for i in range( len(diffs_european) ):
            indx1 = european_languages.index( diffs_european.loc[ i , "LangName_1" ] )
            indx2 = european_languages.index( diffs_european.loc[ i , "LangName_2" ] )

            matrx[ indx1 , indx2 ] = diffs_european.loc[ i , 'Similarity' ]**1.5 #2
            matrx[ indx2 , indx1 ] = diffs_european.loc[ i , 'Similarity' ]**1.5 #2

        maxx = max(matrx.reshape(-1)) + 1 # + 3
        for i in range(len(european_languages)):
            matrx[ i,i ] = maxx 

        distansz = 1 / matrx

        weightsz = np.where( matrx < 0 , 0.0 , matrx )
        for i in range(len(european_languages)):
            weightsz[ i,i ] = 0.0
        
        np.save( "distansz.npy" , distansz )
        np.save( "weightsz.npy" , weightsz )

        np.savetxt( "distansz.csv" , distansz , delimiter=',')
        np.savetxt( "weightsz.csv" , weightsz , delimiter=',')

        with open("euro_names.pkl","wb") as f:
            pkl.dump( european_languages , f )
    getLanguagesAndDistances()
else:
    distansz = np.load( "distansz.npy" ,  )
    weightsz = np.load( "weightsz.npy" )

    with open("euro_names.pkl","rb") as f:
        european_languages = pkl.load( f )

if False:
    def findPoints( distansz, weightsz, european_languages ):

        num_categories = len( european_languages )
        distance_matrix = distansz.copy()
        weight_matrix = weightsz.copy()
        
        def reshape_to_points(points_flat):
            return points_flat.reshape((num_categories, 2))

        def objective_function(points_flat, distance_matrix, weight_matrix):
            if False:
                points_flat = initial_points.copy()
            points = reshape_to_points(points_flat)

            pointsE = points[:, np.newaxis,:] # pointsE.shape
            
            orientation1 = np.tile( pointsE , (1,len(points),1))
            orientation2 = orientation1.transpose( (1,0,2))
            
            diff = orientation1 - orientation2

            fullDistance = np.linalg.norm( diff , axis=2 )

            incorrectDistWeighted = np.abs( distance_matrix - fullDistance )*weight_matrix
            problemm = np.sum(incorrectDistWeighted.reshape(-1))

            return problemm
        
        def constraint_centered( points_flat ):
            points = reshape_to_points(points_flat)
            return [ np.sum(points[:,0]) , np.sum(points[:,1]) ]
        
        if False: #gpt code
            # def objective_function(points_flat, distance_matrix, weight_matrix):
            #     points = reshape_to_points(points_flat)
            #     actual_distances = np.linalg.norm(points[:, np.newaxis, :] - points, axis=2)
            #     diff = (distance_matrix - actual_distances) * weight_matrix
            #     return np.sum(diff**2)

            # Constraint: each entry in the flattened points array should be between 0 and 1
            # constraints = (
            #     {'type': 'ineq', 'fun': lambda x:  x[i] - 1 for i in range(len(initial_points))},
            #     {'type': 'ineq', 'fun': lambda x: -x[i]     for i in range(len(initial_points))})
            pass

        constraints = ( {'type': 'eq', 'fun': constraint_centered } )

        initial_points = np.random.rand(num_categories, 2).flatten()
        startFailure = objective_function( initial_points , distance_matrix, weight_matrix)

        result = minimize(
                        objective_function, 
                        initial_points, 
                        args=(distance_matrix, weight_matrix),
                        constraints=constraints, 
                        #method='SLSQP',
                        #method='Nelder-Mead',
                        #method='trust-constr',
                        #method='Powell',
                        options={'maxiter': 100000} )
        
        optimized_points = reshape_to_points(result.x)
        finalSuccess = objective_function( optimized_points.flatten() , distance_matrix, weight_matrix)

        return optimized_points , startFailure , finalSuccess
    
    candidatePointsz = []
    startFailuresz = []
    finalSuccessz = []
    
    for i in range( 30 ):
        optimized_points , startFailure , finalSuccess = findPoints( distansz, weightsz, european_languages )
        
        candidatePointsz.append( optimized_points )
        startFailuresz.append( startFailure )
        finalSuccessz.append( finalSuccess )

        print( "i:",i, "-", startFailuresz[i],"->",
              finalSuccessz[i],"of", min(finalSuccessz) )

    for i in range( len(startFailuresz) ): #50):
        print( startFailuresz[i],"->",finalSuccessz[i])
    
    bestVal = finalSuccessz[0]
    bestID = 0
    for i in range( len(startFailuresz) ): #50):
        if finalSuccessz[i] < bestVal:
            bestID = i
            bestVal = finalSuccessz[i]
    bestPoints = candidatePointsz[ bestID ]

    #objective_function( bestPoints.flatten() , distance_matrix, weight_matrix)

    np.save( "bestPoints.npy" , bestPoints )

if False: #simply reevaluating
    chosenPoints = np.load( "bestPoints.npy"  )
    distance_matrix = distansz.copy()
    weight_matrix = weightsz.copy()
    def objective_function(points_flat, distance_matrix, weight_matrix):
        if False:
            points_flat = initial_points.copy()
        points = reshape_to_points(points_flat)

        pointsE = points[:, np.newaxis,:] # pointsE.shape
        
        orientation1 = np.tile( pointsE , (1,len(points),1))
        orientation2 = orientation1.transpose( (1,0,2))
        
        diff = orientation1 - orientation2

        fullDistance = np.linalg.norm( diff , axis=2 )

        incorrectDistWeighted = np.abs( distance_matrix - fullDistance )*weight_matrix
        problemm = np.sum(incorrectDistWeighted.reshape(-1))

        return problemm
    print("success_level:",
          objective_function( chosenPoints.flatten() , distance_matrix, weight_matrix))

if False: #displaying it   
    chosenPoints = np.load( "bestPoints.npy"  )

    # Sample data (replace this with your actual data)
    points = chosenPoints.copy()
    labels = european_languages.copy()

    # Extract x and y coordinates
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # Create a scatter plot
    fig = go.Figure()

    # Add circles for each point
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(size=15, symbol='circle', color="white",line=dict(color='tan', width=3) ),  
        # Adjust size and color as needed #color='clear'
    ))

    # Add text labels for each point
    for label, x, y in zip(labels, x_coords, y_coords):
        fig.add_annotation(
            go.layout.Annotation(
                x=x+.005,
                y=y,
                text=label,
                showarrow=False,
                font=dict(color='rgb(0,0,120)', 
                          size=12,
                          family="Century Gothic"),
                xanchor = "left",
                align="left",
            )
        )

    # Set layout properties
    fig.update_layout(
        #title='language map',
       
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        plot_bgcolor='white',  # Set background color to white

        margin=dict(l=10, r=10, b=10, t=80),
        height=800*2.5+70,  # Set the height of the plot
        width=800*2.5,   # '<b>Language Map</b>'

        title=dict(text='Language Map', 
                   font=dict(family="Century Gothic", 
                    size=30, 
                    color='rgb(0,0,120)',),x=.5,),
        
    )

    fig.write_image('language_map.png')
    # Show the plot
    fig.show()