import numpy as np


col_attribute_groups = {
    
    # Biking & Pedaling
    'biking': 'Cycling',
    'pedaling': 'Cycling',

    # Boating & Water Sports
    'rafting': 'WaterSports', 
    'rowing': 'WaterSports',
    
    
    # Descending Movements
    'falling': 'Descending',
    'plummeting': 'Descending',


    # flinging
    'hurling': 'Flinging',
    'pitching': 'Flinging',    


    # Other Physical Actions
    'jumping': 'Jumping',
    'leaping': 'Jumping',
    
    
    # Flying & Circling Movements
    'flapping': 'Airborne', #mainly about birds
    'swooping': 'Airborne', #mainly about birds


    # Sports on Land
    'running': 'Running',
    'sprinting': 'Running',
    'jogging': 'Running',
    
    
    # Farming & Gardening
    'gardening': 'Farming',
    'hoeing': 'Farming',
    'planting': 'Farming',

    # Hiking & Walking
    'hiking': 'Walking',
    'walking': 'Walking',


    # Diving & Submerging
    'submerging': 'Submersion',
    'immersing': 'Submersion',

    # Landing & Touching Down
    'landing': 'Landing', #mostly related to balloons and airplane landing
    'taxiing': 'Landing', #mostly related to airplane landing

    
}


np.save('merged_col_attrs.npy', col_attribute_groups)
