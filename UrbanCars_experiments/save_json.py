import numpy as np


row_attribute_groups = {
    # Humans and People-Related
    'man': 'Human',
    'woman': 'Human',
    'girl': 'Human',
    'person': 'Human',
    'people': 'Human',
    'family': 'Human',

    
    # People
    'crowd': 'People',
    'team': 'People',
    'conference': 'People',
    'festival': 'People',
    'meeting': 'People',
    
    # Indoor Living Spaces
    'living_room': 'Indoor',
    'dining_room': 'Indoor',
    'kitchen': 'Indoor',
    'bathroom': 'Indoor',
    'toilet': 'Indoor',
    'bedroom': 'Indoor',
    'bed': 'Indoor',
    'family_room': 'Indoor',
    'table': 'Indoor',
    'sofa': 'Indoor',
    'desk': 'Indoor',
    'chair': 'Indoor',
    'cabinet': 'Indoor',
    'cupboard': 'Indoor',
    'countertop': 'Indoor',    
    'indoor': 'Indoor', 
    'room': 'Indoor', 
    'inside': 'Indoor',
    'hallway': 'Indoor', 
    'studio': 'Indoor', 
    'stairs': 'Indoor', 
    'stair': 'Indoor', 
    'stairwell': 'Indoor', 
    'stairway': 'Indoor', 
    'rug': 'Indoor', 
    'carpet': 'Indoor', 
    'red_carpet': 'Indoor', 
    'mat': 'Indoor', 
    'hall': 'Indoor',
    'classroom': 'Indoor',
    'office': 'Indoor',

    # Oceans and Seas
    'ocean': 'Seas',
    'sea': 'Seas',
    'reef': 'Seas',
    'wave': 'Seas',
    'ocean_floor': 'Seas',
    'body_of_water': 'Seas',
    'water': 'Seas',

    
    # Rivers and Streams
    'river': 'Seas',
    'stream': 'Seas',
    'brook': 'Seas',
    'riverbank': 'Seas',
    'riverbed': 'Seas',
    'canal': 'Seas',
    'waterfall': 'Seas',
    'rapids': 'Seas',
    'white_water': 'Seas',
    'inlet': 'Seas',
    'flood': 'Seas',
    
    # Lakes and Ponds
    'lake': 'Seas',
    'pond': 'Seas',
    'lagoon': 'Seas',
    'pool': 'Seas',
    'swimming_pool': 'Seas', 
    'waterbody': 'Seas',
    
    # Watercraft and Shorelines
    'shore': 'Seas',
    'pier': 'Seas',
    'beach': 'Seas',
    'island': 'Seas',
    'seashore': 'Seas',
    'beach': 'Seas',
    'coast': 'Seas',
    'coastline': 'Seas',
    #'sand': 'Seas',
    'breakwater': 'Seas',
    'shoreline': 'Seas',
    'seaside': 'Seas',
    'estuary': 'Seas',
    'bay': 'Seas',
    'lakeside': 'Seas',
    'dock': 'Seas',

    # Sea Vessels
    'boat': 'Seas',
    'ship': 'Seas',
    'vessel': 'Seas',
    'fishing_boat': 'Seas',
    'sailboat': 'Seas',
    'kayak': 'Seas',
    'canoe': 'Seas',
    
    # Mountains and Landforms
    'mountain': 'Mountain',
    'hill': 'Mountain',
    'ridge': 'Mountain',
    'canyon': 'Mountain',
    'valley': 'Mountain',
    'cliff': 'Mountain',
    'rock': 'Mountain',
    'peak': 'Mountain',
    'slope': 'Mountain',
    'mesa': 'Mountain',
    'butte': 'Mountain',
    'boulder': 'Mountain',
    'crag': 'Mountain',
    'mound': 'Mountain',
    'rock': 'Mountain',
    'stone': 'Mountain',
    'plateau': 'Mountain',
    'hillside': 'Mountain',
    
    
    # Cave
    'cave': 'Cave',
    'cavern': 'Cave',        
        
    # Deserts and Arid Regions
    'desert': 'Desert',
    'dune': 'Desert',
    'savanna': 'Desert',
    'plain': 'Desert',
    'tundra': 'Desert',
    
    # Genral Land
    'land': 'Desert',
    'Earth': 'Desert',
    'earth': 'Desert',
    'soil': 'Desert',
    'sand': 'Desert',
    'mud': 'Desert',
    'dirt': 'Desert',
    
    # Forest and Vegetation Areas
    'forest': 'Forest',
    'jungle': 'Forest',
    'grassland': 'Forest',
    'grass': 'Forest',
    'flower': 'Forest',
    'swamp': 'Forest',
    'scrubland': 'Forest',
    'prairie': 'Forest',
    'meadow': 'Forest',
    'tree': 'Forest',
    'leaf': 'Forest',
    
    # Farms and Rural Areas
    'farm': 'Forest',
    'pasture': 'Forest',
    'barn': 'Forest',
    'vineyard': 'Forest',
    'field': 'Forest',
    'farm': 'Forest',
    'ranch': 'Forest',
    'pasture': 'Forest',
    'barn': 'Forest',
    'cornfield': 'Forest',
    'paddock': 'Forest',
    

    
    # Outdoor Leisure and Recreational Areas
    'park': 'Forest',
    'garden': 'Forest',
    'lawn': 'Forest',
    'front_yard': 'Forest',
    'back_yard': 'Forest',
    'backyard': 'Forest',
    'yard': 'Forest',
    'playground': 'Forest',
    'theme_park': 'Forest',
    'amusement_park': 'Forest',
    'campsite': 'Forest',
    'campus': 'Forest',
    'boot_camp': 'Forest',

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
    'house': 'Buildings',
    'village': 'Buildings',
    'apartment_building': 'Buildings',
    'home': 'Buildings',
    'dwelling': 'Buildings',
    'apartment': 'Buildings',
    'residential_district': 'Buildings',
    'neighborhood': 'Buildings',
    'town': 'Buildings',
    'suburb': 'Buildings',

    # Urban Structures and Landmarks
    'skyscraper': 'Buildings',
    'fountain': 'Buildings',
    'public_square': 'Buildings',
    'town_hall': 'Buildings',
    'city_hall': 'Buildings',
    'clock_tower': 'Buildings',
    'urban_area': 'Buildings',
    'building': 'Buildings',
    'urban': 'Buildings',
    'city': 'Buildings',
    
    # Industrial and Mechanical Locations
    'factory': 'Buildings',
    'warehouse': 'Buildings',
    'construction': 'Buildings',
    'workshop': 'Buildings',
    'garage': 'Buildings',

    # Office and Institutional Buildings
    'office_building': 'Buildings',
    'hospital': 'Buildings',
    'school': 'Buildings',
    'clinic': 'Buildings',
    
    # Forge
    'forge': 'Forge',
    'foundry': 'Forge',
    'fire': 'Forge',
    
    
    # General Outdoor 
    'outdoor': 'Outside',
    'outside': 'Outside',
    'outdoors': 'Outside',  
    'parade': 'Outside',  
    
    
    

    # Roads and Pathways
    'sidewalk': 'Roads',
    'street': 'Roads',
    'trail': 'Roads',
    'way': 'Roads',
    'path': 'Roads',
    'driveway': 'Roads',
    'road': 'Roads',
    'roadway': 'Roads',
    'highway': 'Roads',
    'local_road': 'Roads',
    'expressway': 'Roads',
    'dirt_track': 'Roads',
    'racetrack': 'Roads',
    'track': 'Roads',
    'alley': 'Roads',
    'pathway': 'Roads',
    'tarmacadam': 'Roads',
    'parking_lot': 'Roads',
    
    # Snow and Ice Environments
    'snow': 'Snow',
    'snowfield': 'Snow',
    'snowbank': 'Snow',
    'ice': 'Snow',
    'ice_field': 'Snow',
    'ice_rink': 'Snow',
    'ski_slope': 'Snow',
    'glacier': 'Snow',



    # Retail and Commerce
    'shop': 'Shop',
    'package_store': 'Shop',
    'market': 'Shop',
    'store': 'Shop',
    'grocery': 'Shop',
    'mall': 'Shop',
    'shopping_center': 'Shop',
    'store': 'Shop',
    'convenience_store': 'Shop',
    'shopping_center': 'Shop',
    'grocery_store': 'Shop',
    
    # Food and Beverage Establishments
    'restaurant': 'Restaurant',
    'bar': 'Restaurant',
    'cafe': 'Restaurant',
    'pub': 'Restaurant',
    'coffee_shop': 'Restaurant',
    'bakery': 'Restaurant',    

    # Space and Astronomical
    'outer_space': 'Space',
    'planet': 'Space',
    'moon': 'Space',
    'galaxy': 'Space',
    'satellite': 'Space',
    'astronaut': 'Space',
    'star': 'Space',
    'space': 'Space',    
    
    # Sky
    'sky': 'Sky',
    'cloud': 'Sky',
    'blue_sky': 'Sky',
    'mid_air': 'Sky',
    'air': 'Sky',
    
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
    'cinema': 'Public Venues',
    
    # Sports Venues and Fields
    'stadium': 'Sports Venues',
    'court': 'Sports Venues',
    'basketball_court': 'Sports Venues',
    'tennis_court': 'Sports Venues',
    'volleyball_court': 'Sports Venues',
    'football_field': 'Sports Venues',
    'football_stadium': 'Sports Venues',
    'playing_field': 'Sports Venues',
    'ball_field': 'Sports Venues',
    
    # Hockey
    'rink': 'Rink',
    'ice_hockey': 'Rink',
    'ice_hockey_rink': 'Rink',
    
    # Ring
    'ring': 'Ring',
    'wrestling': 'Ring',
    'wrestling_ring': 'Ring',
    'boxing_ring': 'Ring',
    
    # Land Vehicles
    'car': 'Vehicles',
    'truck': 'Vehicles',
    'motorcycle': 'Vehicles',
    'bus': 'Vehicles',
    'streetcar': 'Vehicles',
    
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
    'church': 'Religious Sites',
    'temple': 'Religious Sites',
    'monument': 'Religious Sites',
    'statue': 'Religious Sites',
    'memorial': 'Religious Sites',
    'shrine': 'Religious Sites',
    
}


np.save('merged_row_attrs.npy', row_attribute_groups)
