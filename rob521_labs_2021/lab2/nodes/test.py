import yaml

with open('../maps/willowgarageworld_05res.yaml', 'r') as file:
    mapdata = yaml.safe_load(file)
    print(mapdata['origin'])
    print(mapdata['resolution'])