import json

class_names = [
    "Alfalfa",
    "Asparagus",
    "BlueVervain",
    "BroadleafPlantain",
    "BullThistle",
    "Cattail",
    "Chickweed",
    "Chicory",
    "Cleavers",
    "Coltsfoot",
    "CommonSowThistle",
    "CommonYarrow",
    "Coneflower",
    "CreepingCharlie",
    "CrimsonClover",
    "CurlyDock",
    "DaisyFleabane",
    "Dandelion",
    "DownyYellowViolet",
    "Elderberry",
    "EveningPrimrose",
    "FernLeafYarrow",
    "FieldPennycress",
    "Fireweed",
    "ForgetMeNot",
    "GarlicMustard",
    "Harebell",
    "Henbit",
    "HerbRobert",
    "JapaneseKnotweed",
    "JoePyeWeed",
    "Knapweed",
    "Kudzu",
    "LambsQuarters",
    "Mallow",
    "Mayapple",
    "Meadowsweet",
    "MilkThistle",
    "Mullein",
    "NewEnglandAster",
    "Partridgeberry",
    "Peppergrass",
    "Pickerelweed",
    "PineappleWeed",
    "PricklyPearCactus",
    "PurpleDeadnettle",
    "QueenAnnesLace",
    "RedClover",
    "SheepSorrel",
    "ShepherdsPurse",
    "SpringBeauty",
    "Sunflower",
    "SupplejackVine",
    "TeaPlant",
    "Teasel",
    "Toothwort",
    "Vervain Mallow",
    "WildBeeBalm",
    "WildBlackCherry",
    "WildGrapeVine",
    "WildLeek",
    "WoodSorrel"
]

with open('class_names.json', 'w') as f:
    json.dump(class_names, f)