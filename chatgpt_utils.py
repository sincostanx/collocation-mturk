import json
from PIL import Image
import pandas as pd
from utils import image_grid
import inflect
from nltk.corpus import words
import zipfile
import os
from pathlib import Path
import numpy as np
import itertools

PROMPT_ADD_VER2 = """
Instructions:

Suggest 5 generic items that can plausibly appear in this image. Here are some guidelines you MUST follow:
1. Focus on items that are suitable for the setting shown in the image. You can be creative here as long as it makes sense semantically, such as putting large-enough sunglasses on a sphinx. 
2. Pay close attention to the available space and surfaces in the image. Consider where the item can be realistically placed, worn, or used within the context of the image. Items that require surfaces (e.g., tables, shelves) should only be selected if such surfaces are visible in the image. This implies that a vase shouldn't be selected given an image of a wall.
3. Personal accessories or clothing items (e.g., hats, sunglasses, backpacks) are allowed as long as they could naturally be worn or carried by the person in the image without drastically altering the composition or focus. Avoid selecting items that require additional space or surfaces not visible in the image. Only choose items that can be realistically placed or used within the context shown. For example, if the image is a close-up of a chair, do not select items like a coffee cup or anything that requires a table, assuming there might be a table nearby.
4. You may select items that naturally fit in the sky or background of the scene, even if they are not directly interacting with the visible surfaces. This includes elements like birds, airplanes, or clouds that complement the broader environment of the image.
5. Do not select items already present in the image, including both tangible objects (e.g., a hat) and intangible elements (e.g., sunlight rays, water droplets, mist). For example, if the image shows a person wearing a hat, do not select another hat, and if there are water droplets in the image, do not suggest adding more.
6. Avoid adding objects that would drastically alter the focus or composition of the image. However, you may add smaller, more subtle items like birds, balloons, or other elements that naturally integrate into the scene without overshadowing the primary focus. These additions should complement the existing composition and maintain the visual balance of the image.
7. Consider the scale, proportion, and visibility of the items relative to the context and camera angle shown in the image. Ensure that the items chosen do not overwhelm the space, are not too small to be noticeable, or are too large for the visible space.
8. Do not select intangible items such as sunlight ray and rainbow.
9. After selecting items, provide only the names of the items in your response, separated by commas. Do not include any additional descriptions or specify where the items should appear.

Examples:
1. If the image shows a table in a living room, you can select an apple to place on the table, but not an airplane.
2. If the image includes a table, do not select any additional tables.
3. If the image is a close-up of a chair, do not assume that there might be a table nearby and select items like a coffee cup or anything that requires a table.
"""

PROMPT_ADD_VER1 = """
Instructions:

Suggest 5 generic items that can plausibly appear in this image. Here are some guidelines you MUST follow:
1. Focus on items that are suitable for the setting shown in the image. You can be creative here as long as it makes sense semantically, such as putting large-enough sunglasses on a sphinx. 
2. Pay close attention to the available space and surfaces in the image. Consider where the item can be realistically placed or used within the context of the image. Items that require surfaces (e.g., tables, shelves) should only be selected if such surfaces are visible in the image. This implies that a vase shouldn't be selected given an image of a wall.
3. Avoid selecting items that require additional space or surfaces not visible in the image. Only choose items that can be realistically placed or used within the context shown. For example, if the image is a close-up of a chair, do not select items like a coffee cup or anything that requires a table, assuming there might be a table nearby.
4. You may select items that naturally fit in the sky or background of the scene, even if they are not directly interacting with the visible surfaces. This includes elements like birds, airplanes, or clouds that complement the broader environment of the image.
5. Do not select items already present in the image, including both tangible objects (e.g., a hat) and intangible elements (e.g., sunlight rays, water droplets, mist). For example, if the image shows a person wearing a hat, do not select another hat, and if there are water droplets in the image, do not suggest adding more.
6. Avoid adding objects that would drastically alter the focus or composition of the image. However, you may add smaller, more subtle items like birds, balloons, or other elements that naturally integrate into the scene without overshadowing the primary focus. These additions should complement the existing composition and maintain the visual balance of the image.
7. Consider the scale, proportion, and visibility of the items relative to the context and camera angle shown in the image. Ensure that the items chosen do not overwhelm the space, are not too small to be noticeable, or are too large for the visible space.
8. Do not select intangible items such as sunlight ray and rainbow.
9. After selecting items, provide only the names of the items in your response, separated by commas. Do not include any additional descriptions or specify where the items should appear.

Examples:
1. If the image shows a table in a living room, you can select an apple to place on the table, but not an airplane.
2. If the image includes a table, do not select any additional tables.
3. If the image is a close-up of a chair, do not assume that there might be a table nearby and select items like a coffee cup or anything that requires a table.
"""

PROMPT_SELECT_VER1 = """
    Instructions:

    Choose 5 items from the given list that best fit the context of the image. Here are some guidelines you should follow:
    1. Focus on items that are suitable for the setting shown in the image. You can be creative here as long as it makes sense semantically, such as putting large-enough sunglasses on a sphinx. 
    2. Pay close attention to the available space and surfaces in the image. Consider where the item can be realistically placed or used within the context of the image. Items that require surfaces (e.g., tables, shelves) should only be selected if such surfaces are visible in the image. This implies that a vase shouldn't be selected given an image of a wall.
    3. Avoid selecting items that require additional space or surfaces not visible in the image. Only choose items that can be realistically placed or used within the context shown. For example, if the image is a close-up of a chair, do not select items like a coffee cup or anything that requires a table, assuming there might be a table nearby.
    4. Do not select items already present in the image. For example, if the image shows a person wearing a hat, do not select another hat.
    5. Consider the scale, proportion, and visibility of the items relative to the context and camera angle shown in the image. Ensure that the items chosen do not overwhelm the space, are not too small to be noticeable, or are too large for the visible space.
    6. After selecting items, reply with their names separated by comma.

    Examples:
    1. If the image shows a table in a living room, you can select an apple to place on the table, but not an airplane.
    2. If the image includes a table, do not select any additional tables.
    3. If the image is a close-up of a chair, do not assume that there might be a table nearby and select items like a coffee cup or anything that requires a table.

    List of items:

    accordion
    adhesive tape
    air conditioner
    airplane
    alarm clock
    alcohol
    almond
    alpaca
    ambulance
    apple
    apron
    armadillo
    armchair
    artichoke
    asparagus
    avocado
    awning
    axe
    backpack
    bagel
    balance beam
    ball
    balloon
    banana
    band-aid
    banjo
    banner
    barge
    barrel
    baseball
    baseball bat
    baseball cap
    baseball glove
    basket
    bat (animal)
    bath towel
    bathroom cabinet
    bathtub
    bead
    beaker
    bean curd
    beanie
    bear
    bed
    bell pepper
    belt
    bench
    bicycle
    bicycle helmet
    bicycle wheel
    bidet
    billboard
    billiard table
    binoculars
    bird
    blanket
    blender
    blinker
    blueberry
    boat
    bolt
    bomb
    book
    bookcase
    boot
    bottle
    bottle cap
    bottle opener
    bow (decorative ribbons)
    bowl
    bowling equipment
    box
    boy
    bracelet
    brake light
    bread
    briefcase
    broccoli
    bronze sculpture
    brown bear
    bucket
    bull
    bun
    buoy
    burrito
    bus (vehicle)
    bust
    butterfly
    button
    cabbage
    cabinet
    cabinetry
    cake
    cake stand
    calculator
    camel
    camera
    can
    can opener
    canary
    candle
    candy
    cannon
    canoe
    car (automobile)
    carrot
    cart
    cassette deck
    castle
    cat
    cattle
    ceiling fan
    celery
    cello
    cellular telephone
    chainsaw
    chair
    cheese
    cheetah
    cherry
    chest of drawers
    chicken
    chime
    chisel
    choker
    chopping board
    chopsticks
    christmas tree
    cistern
    clock
    clock tower
    closet
    coat
    coat hanger
    cocktail
    cocktail shaker
    coconut
    coffee (drink)
    coffee cup
    coffee table
    coffeemaker
    coin
    common fig
    common sunflower
    computer keyboard
    computer monitor
    computer mouse
    condiment
    cone
    convenience store
    cookie
    cooking spray
    cooler (for food)
    corded phone
    countertop
    cow
    cowboy hat
    crab
    cracker
    crate
    cream
    cricket ball
    crisp (potato chip)
    crocodile
    crossbar
    crown
    crumb
    crutch
    cucumber
    cup
    cupboard
    cupcake
    curtain
    cushion
    cutting board
    dagger
    deck chair
    deer
    desk
    diaper
    dice
    digital clock
    dining table
    dinosaur
    dishwasher
    dog
    dog bed
    dog collar
    doll
    dolphin
    door
    door handle
    doorknob
    doughnut
    dragonfly
    drawer
    dress
    drill (tool)
    drum
    duck
    dumbbell
    eagle
    earphone
    earring
    edible corn
    egg
    envelope
    eraser
    face powder
    facial tissue holder
    falcon
    fan
    faucet
    fax
    fedora
    figurine
    filing cabinet
    fireplace
    fixed-wing aircraft
    flag
    flagpole
    flamingo
    flashlight
    flip-flop (sandal)
    flower arrangement
    flowerpot
    flute
    flying disc
    food processor
    football
    football helmet
    fork
    fountain
    fox
    french fries
    french horn
    frog
    frying pan
    garlic
    gas stove
    giraffe
    girl
    glass (drink container)
    glasses
    glove
    goat
    goggles
    goldfish
    golf ball
    golf cart
    gondola
    goose
    grape
    grapefruit
    green bean
    green onion
    grinder
    guacamole
    guitar
    gull
    hair dryer
    hair spray
    ham
    hamburger
    hammer
    hamster
    hand dryer
    handbag
    handgun
    handle
    harbor seal
    harmonica
    harp
    harpsichord
    hat
    headband
    headboard
    headlight
    headphones
    heater
    hedgehog
    helicopter
    helmet
    high heels
    hiking equipment
    hinge
    hippopotamus
    honeycomb
    hook
    horizontal bar
    horse
    hose
    hot dog
    house
    humidifier
    ice cream
    indoor rower
    ipod
    jacket
    jacuzzi
    jaguar (animal)
    jar
    jeans
    jellyfish
    jersey
    jet ski
    jug
    juice
    kangaroo
    kettle
    key
    kitchen & dining room table
    kitchen knife
    kitchen sink
    kite
    kiwi fruit
    knee pad
    knife
    knob
    koala
    ladder
    ladle
    ladybug
    lamp
    lamppost
    lampshade
    lanyard
    laptop
    latch
    lego
    legume
    lemon
    leopard
    lettuce
    license plate
    light bulb
    light switch
    lighthouse
    lily
    lime
    limousine
    lion
    lipstick
    lizard
    lobster
    log
    loveseat
    lynx
    magazine
    magnet
    magpie
    man
    mandarin orange
    mango
    maple
    maraca
    mask
    mast
    measuring cup
    mechanical fan
    microphone
    microwave oven
    milk
    miniskirt
    minivan
    mirror
    missile
    mixer
    mixing bowl
    mobile phone
    monkey
    motor
    motor scooter
    motorcycle
    muffin
    mug
    mule
    mushroom
    musical keyboard
    nail (construction)
    napkin
    necklace
    necktie
    newspaper
    nightstand
    oboe
    office building
    onion
    orange (fruit)
    organ (musical instrument)
    ostrich
    otter
    oven
    owl
    oyster
    paddle
    painting
    palm tree
    pan (for cooking)
    pancake
    panda
    paper cutter
    paper plate
    paper towel
    parachute
    parking meter
    parrot
    pasta
    pastry
    pea (food)
    peach
    pear
    pen
    pencil case
    pencil sharpener
    penguin
    perfume
    person
    personal flotation device
    piano
    pickle
    pickup truck
    picnic basket
    picture frame
    pig
    pigeon
    pillow
    pineapple
    pipe
    pitcher (vessel for liquid)
    pizza
    pizza cutter
    place mat
    plastic bag
    plate
    platter
    polar bear
    pole
    polo shirt
    pomegranate
    pop (soda)
    popcorn
    porch
    porcupine
    poster
    pot
    potato
    power plugs and sockets
    prawn
    pressure cooker
    pretzel
    printer
    propeller
    pumpkin
    punching bag
    rabbit
    raccoon
    radish
    railcar (part of a train)
    raspberry
    ratchet (device)
    raven
    rays and skates
    rearview mirror
    red panda
    reflector
    refrigerator
    rhinoceros
    rifle
    ring
    ring binder
    rocket
    roller skates
    rose
    rugby ball
    ruler
    saddle (on an animal)
    saddle blanket
    sail
    salad
    salt and pepper shakers
    sandal (type of shoe)
    sandwich
    saucer
    sausage
    saxophone
    scale
    scarf
    scissors
    scoreboard
    scorpion
    screwdriver
    sea lion
    sea turtle
    seahorse
    seat belt
    segway
    serving tray
    sewing machine
    shark
    sheep
    shelf
    shirt
    shoe
    shorts
    shotgun
    shower
    shrimp
    signboard
    sink
    skateboard
    ski
    ski boot
    ski parka
    ski pole
    skirt
    skunk
    skyscraper
    slow cooker
    snail
    snake
    snowboard
    snowman
    snowmobile
    snowplow
    soap
    soap dispenser
    soccer ball
    sock
    sofa
    sofa bed
    sombrero
    sparrow
    spatula
    speaker (stereo equipment)
    spectacles
    spice rack
    spider
    spoon
    sports uniform
    squid
    squirrel
    stairs
    stapler
    starfish
    stationary bicycle
    statue (sculpture)
    steering wheel
    stethoscope
    stool
    stop sign
    stove
    strap
    straw (for drinking)
    strawberry
    street light
    street sign
    stretcher
    studio couch
    submarine
    suit
    suitcase
    sun hat
    sunglasses
    surfboard
    sushi
    swan
    sweater
    sweatshirt
    swim cap
    swimming pool
    swimsuit
    swimwear
    sword
    syringe
    table
    table tennis racket
    tablecloth
    tablet computer
    taco
    tag
    taillight
    tank
    tank top (clothing)
    tap
    tarp
    tart
    taxi
    tea
    teapot
    teddy bear
    telephone
    telephone pole
    television
    tennis ball
    tennis racket
    tent
    tiara
    tick
    tie
    tiger
    tin can
    tire
    toaster
    toilet
    toilet paper
    toothbrush
    toothpick
    torch
    tortoise
    towel
    towel rack
    tower
"""

PROMPT_OLDEST = """
Answer the following questions using the given image. List the objects' names for the 1st and 2nd questions in the 1st and 2nd lines (separated by commas) without explanations. Each object should be fully visible after editing and not already included in the original image.

1. Suggest 5 objects that can be edited into this image and placed in only one location
2. Suggest 5 objects that can be edited into this image and placed in multiple locations, but not just anywhere

Answer 'None' if it is implausible to insert any objects.
"""

class TextParser:
    def __init__(self):
        self.word_list = set(words.words())

    def is_meaningful_word(self, word):
        if len(word) == 1 and word.lower() not in ['a', 'i']:
            return False
        return word.lower() in self.word_list 
    
    def is_valid_phrase(self, phrase):
        word_components = phrase.split()
        return all(self.is_meaningful_word(word) for word in word_components)
    
    def remove_last_dot(self, s):
        index = s.rfind('.')
        return s[:index] + s[index+1:] if index != 1 else s
    
    def plural_to_singular(self, word):
        p = inflect.engine()
        singular = p.singular_noun(word)
        return singular if singular else word
    
    def reorder_list_with_none_at_end(self, lst):
        non_none_elements = [x for x in lst if x is not None]
        none_elements = [x for x in lst if x is None]
        return non_none_elements + none_elements
    
    def __call__(self, text):
        text = text.split('\n')
        text = [val for val in text if val]

        x = text[0][3:].lower().split(', ')
        y = text[1][3:].lower().split(', ')

        x = [self.remove_last_dot(val) for val in x]
        y = [self.remove_last_dot(val) for val in y]

        if (len(x) == 1) and (x[0] == "none"): x = [None] * 5
        if (len(y) == 1) and (y[0] == "none"): y = [None] * 5

        if (len(x) != 5) or (len(y) != 5):
            print(len(x), len(y), text)
            x = [None] * 5
            y = [None] * 5
        else:
            if (len(x) == 5) and (x[0] is not None): x = [self.plural_to_singular(word) for word in x]
            if (len(y) == 5) and (y[0] is not None): y = [self.plural_to_singular(word) for word in y]

            check_valid = lambda x: x if self.is_valid_phrase(x) else None
            if (len(x) == 5) and (x[0] is not None): x = [check_valid(word) for word in x]
            if (len(y) == 5) and (y[0] is not None): y = [check_valid(word) for word in y]

            x = self.reorder_list_with_none_at_end(x)
            y = self.reorder_list_with_none_at_end(y)

        return x + y

# def extract_data(outputs):
#     df = []
#     for output in outputs:
#         df.append({
#             "id": output["id"],
#             "custom_id": output["custom_id"],
#             "status_code": output["response"]["status_code"],
#             "model": output["response"]["body"]["model"],
#             "prompt_tokens": output["response"]["body"]["usage"]["prompt_tokens"],
#             "completion_tokens": output["response"]["body"]["usage"]["completion_tokens"],
#             "answer": output["response"]["body"]["choices"][0]["message"]["content"],
#         })

#     return pd.DataFrame(df)

# def load_as_dataframe(paths):
#     outputs = []
#     for path in paths:
#         outputs += ([json.loads(i) for i in open(path).readlines()])

#     df = extract_data(outputs)
#     return df