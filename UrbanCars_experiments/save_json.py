import numpy as np


attribute_groups = {
    # Humans and People-Related
    'man': 'Human',
    'woman': 'Human',
    'girl': 'Human',
    'person': 'Human',
    'people': 'Human',
    'family': 'Human',
    'team': 'Human',

    # Indoor Living Spaces
    'living_room': 'Indoor',
    'dining_room': 'Indoor',
    'kitchen': 'Indoor',
    'bathroom': 'Indoor',
    'toilet': 'Indoor',
    'bedroom': 'Indoor',
    'family_room': 'Indoor',
    'table': 'Indoor',
    'sofa': 'Indoor',
    'desk': 'Indoor',
    'chair': 'Indoor',
    'cabinet': 'Indoor',
    'cupboard': 'Indoor',
    'countertop': 'Indoor',    
    'indoor': 'Indoor', 
    
    # Office and Institutional Buildings
    'office': 'Office and Institutional',
    'hospital': 'Office and Institutional',
    'school': 'Office and Institutional',
    'clinic': 'Office and Institutional',


    # Oceans and Seas
    'ocean': 'Oceans and Seas',
    'sea': 'Oceans and Seas',
    'reef': 'Oceans and Seas',
    'wave': 'Oceans and Seas',
    'ocean_floor': 'Oceans and Seas',
    'body_of_water': 'Oceans and Seas',

    
    # Rivers and Streams
    'river': 'Rivers and Streams',
    'stream': 'Rivers and Streams',
    'brook': 'Rivers and Streams',
    'riverbank': 'Rivers and Streams',
    'riverbed': 'Rivers and Streams',
    'canal': 'Rivers and Streams',
    'waterfall': 'Rivers and Streams',
    'rapids': 'Rivers and Streams',
    'white_water': 'Rivers and Streams',
    'inlet': 'Rivers and Streams',
    'flood': 'Rivers and Streams',
    
    # Lakes and Ponds
    'lake': 'Lakes and Ponds',
    'pond': 'Lakes and Ponds',
    'lagoon': 'Lakes and Ponds',
    'pool': 'Lakes and Ponds',
    'waterbody': 'Lakes and Ponds',
    
    # Watercraft and Shorelines
    'shore': 'Coastal Areas',
    'pier': 'Coastal Areas',
    'beach': 'Coastal Areas',
    'island': 'Coastal Areas',
    'seashore': 'Coastal Areas',
    'beach': 'Coastal Areas',
    'coast': 'Coastal Areas',
    'coastline': 'Coastal Areas',
    'sand': 'Coastal Areas',
    'breakwater': 'Coastal Areas',
    'shoreline': 'Coastal Areas',
    'seaside': 'Coastal Areas',
    'estuary': 'Coastal Areas',
    'bay': 'Coastal Areas',
    'lakeside': 'Coastal Areas',

    # Sea Vessels
    'boat': 'Sea Vessels',
    'ship': 'Sea Vessels',
    'vessel': 'Sea Vessels',
    'fishing_boat': 'Sea Vessels',
    'sailboat': 'Sea Vessels',
    'kayak': 'Sea Vessels',
    'canoe': 'Sea Vessels',
    
    # Mountains and Landforms
    'mountain': 'Landforms',
    'hill': 'Landforms',
    'ridge': 'Landforms',
    'canyon': 'Landforms',
    'valley': 'Landforms',
    'cliff': 'Landforms',
    'rock': 'Landforms',
    'peak': 'Landforms',
    'slope': 'Landforms',
    'mesa': 'Landforms',
    'butte': 'Landforms',
    'boulder': 'Landforms',
    'crag': 'Landforms',
    'mound': 'Landforms',
    'rock': 'Landforms',
    'stone': 'Landforms',
    'plateau': 'Landforms',

    # Deserts and Arid Regions
    'desert': 'Desert and Arid Regions',
    'dune': 'Desert and Arid Regions',
    'savanna': 'Desert and Arid Regions',
    'plain': 'Desert and Arid Regions',
    'tundra': 'Desert and Arid Regions',

    # Transportation and Infrastructure
    'railway_station': 'Train',
    'metro': 'Train',
    'train': 'Train',
    'train_station': 'Train',
    'subway_station': 'Train',
    'subway': 'Train',
    
    # Airport
    'airport': 'Airport',
    'air_base': 'Airport',
    'air_terminal': 'Airport',
    'airfield': 'Airport',
    'airplane': 'Airport',
    'jet': 'Airport',
    
    # Residential and Public Buildings
    'house': 'Residential and Urban Structures',
    'village': 'Residential and Urban Structures',
    'apartment_building': 'Residential and Urban Structures',
    'home': 'Residential and Urban Structures',
    'dwelling': 'Residential and Urban Structures',
    'apartment': 'Residential and Urban Structures',
    'residential_district': 'Residential and Urban Structures',
    'neighborhood': 'Residential and Urban Structures',
    'town': 'Residential and Urban Structures',
    'suburb': 'Residential and Urban Structures',
    'streetcar': 'Residential and Urban Structures',

    # Urban Structures and Landmarks
    'skyscraper': 'Urban Structures and Landmarks',
    'fountain': 'Urban Structures and Landmarks',
    'public_square': 'Urban Structures and Landmarks',
    'town_hall': 'Urban Structures and Landmarks',
    'city_hall': 'Urban Structures and Landmarks',
    'clock_tower': 'Urban Structures and Landmarks',
    'urban_area': 'Urban Structures and Landmarks',
    'building': 'Urban Structures and Landmarks',
    
    # Industrial and Mechanical Locations
    'factory': 'Industrial and Mechanical',
    'warehouse': 'Industrial and Mechanical',
    'construction': 'Industrial and Mechanical',
    'workshop': 'Industrial and Mechanical',
    'garage': 'Industrial and Mechanical',
    
    
    # Forest and Vegetation Areas
    'forest': 'Forest and Vegetation',
    'jungle': 'Forest and Vegetation',
    'grassland': 'Forest and Vegetation',
    'swamp': 'Forest and Vegetation',
    'scrubland': 'Forest and Vegetation',
    'prairie': 'Forests and Vegetation',
    'meadow': 'Forests and Vegetation',
    
    # Farms and Rural Areas
    'farm': 'Rural and Agricultural',
    'pasture': 'Rural and Agricultural',
    'barn': 'Rural and Agricultural',
    'vineyard': 'Rural and Agricultural',
    'field': 'Rural and Agricultural',
    'farm': 'Rural and Agricultural',
    'ranch': 'Rural and Agricultural',
    'pasture': 'Rural and Agricultural',
    'barn': 'Rural and Agricultural',
    'cornfield': 'Rural and Agricultural',
    'paddock': 'Rural and Agricultural',
    
    # Outdoor Leisure and Recreational Areas
    'park': 'Outdoor Recreational',
    'garden': 'Outdoor Recreational',
    'lawn': 'Outdoor Recreational',
    'front_yard': 'Outdoor Recreational',
    'playground': 'Outdoor Recreational',
    'theme_park': 'Outdoor Recreational',
    'amusement_park': 'Outdoor Recreational',
    'campsite': 'Outdoor Recreational',
    'campus': 'Outdoor Recreational',
    'boot_camp': 'Outdoor Recreational',

    # Roads and Pathways
    'sidewalk': 'Roads and Paths',
    'street': 'Roads and Paths',
    'trail': 'Roads and Paths',
    'way': 'Roads and Paths',
    'path': 'Roads and Paths',
    'driveway': 'Roads and Paths',
    'road': 'Roads and Paths',
    'roadway': 'Roads and Paths',
    'highway': 'Roads and Paths',
    'local_road': 'Roads and Paths',
    'expressway': 'Roads and Paths',
    'dirt_track': 'Roads and Paths',
    'racetrack': 'Roads and Paths',
    'track': 'Roads and Paths',
    'alley': 'Roads and Paths',
    
    # Snow and Ice Environments
    'snow': 'Snow and Ice',
    'snowfield': 'Snow and Ice',
    'snowbank': 'Snow and Ice',
    'ice': 'Snow and Ice',
    'ice_field': 'Snow and Ice',
    'ice_rink': 'Snow and Ice',
    'ski_slope': 'Snow and Ice',
    'glacier': 'Snow and Ice',




    # Retail and Commerce
    'shop': 'Retail and Commerce',
    'package_store': 'Retail and Commerce',
    'market': 'Retail and Commerce',
    'store': 'Retail and Commerce',
    'grocery': 'Retail and Commerce',
    'mall': 'Retail and Commerce',
    'shopping_center': 'Retail and Commerce',
    
    # Food and Beverage Establishments
    'restaurant': 'Food and Beverage Establishments',
    'bar': 'Food and Beverage Establishments',
    'cafe': 'Food and Beverage Establishments',
    'pub': 'Food and Beverage Establishments',
    'coffee_shop': 'Food and Beverage Establishments',

    # Space and Astronomical
    'outer_space': 'Space and Astronomical',
    'planet': 'Space and Astronomical',
    'moon': 'Space and Astronomical',
    'galaxy': 'Space and Astronomical',
    'satellite': 'Space and Astronomical',
    'astronaut': 'Space and Astronomical',
    'star': 'Space and Astronomical',
    

    
    # Public Venues
    'casino': 'Public Venues',
    'theater': 'Public Venues',
    'ballroom': 'Public Venues',
    'concert': 'Public Venues',
    'show': 'Public Venues',
    'auditorium': 'Public Venues',
    'stage': 'Public Venues',
    'opera_house': 'Public Venues',
    'movie': 'Public Venues',
    
    # Sports Venues and Fields
    'stadium': 'Sports Venues',
    'court': 'Sports Venues',
    'basketball_court': 'Sports Venues',
    'tennis_court': 'Sports Venues',
    'volleyball_court': 'Sports Venues',
    'football_field': 'Sports Venues',
    'playing_field': 'Sports Venues',
    'ball_field': 'Sports Venues',
    
    # Land Vehicles
    'car': 'Land Vehicles',
    'truck': 'Land Vehicles',
    'motorcycle': 'Land Vehicles',
    'bus': 'Land Vehicles',
    
    # Competitive Events
    'tournament': 'Competitive Events',
    'competition': 'Competitive Events',
    'contest': 'Competitive Events',
    'race': 'Competitive Events',
    'match': 'Competitive Events',
    'game': 'Competitive Events',
    'Olympic_Games': 'Competitive Events',
    'marathon': 'Competitive Events',
    'obstacle_race': 'Competitive Events',
    'relay': 'Competitive Events',
    'track_meet': 'Competitive Events',
    
    # Religious and Historical Sites
    'church': 'Religious and Historical Sites',
    'temple': 'Religious and Historical Sites',
    'monument': 'Religious and Historical Sites',
    'statue': 'Religious and Historical Sites',
    'memorial': 'Religious and Historical Sites',
    'shrine': 'Religious and Historical Sites',
    
}


np.save('merged_attrs.npy', attribute_groups)
