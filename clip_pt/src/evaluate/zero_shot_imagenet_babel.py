# Scrip adapted from https://github.com/gregor-ge/Babel-ImageNet

import json
import sys
import os

import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize, CenterCrop, ToTensor, Compose, Normalize
from transformers import AutoTokenizer

import argparse
import json
import os
from sklearn.metrics import top_k_accuracy_score, accuracy_score

# Add previous and current path to search for modules
sys.path.append("./")
sys.path.append("../")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.open_CLIP import OpenCLIP
from models.open_CLIP_adapter import OpenCLIPAdapter
from models.open_clip_wrapper import OpenCLIPWrapper


xlmr_langs = [l.upper() for l in ['kn', 'sk', 'et', 'gu', 'sl', 'ka', 'gl', 'hi', 'ja', 'no', 'ms', 'my', 'eo', 'fi', 'ar', 'lv', 'de', 'ha', 'mn', 'sa', 'fr', 'br', 'or', 'ta', 'bs', 'lo', 'he', 'si', 'te', 'es', 'el', 'pt', 'km', 'ro', 'sv', 'bg', 'vi', 'az', 'la', 'th', 'af', 'om', 'eu', 'ga', 'ca', 'nl', 'ps', 'ml', 'uk', 'hy', 'jv', 'gd', 'sd', 'tl', 'zh', 'mk', 'am', 'kk', 'da', 'pa', 'ug', 'sq', 'fy', 'su', 'mg', 'is', 'ku', 'lt', 'yi', 'be', 'uz', 'id', 'sw', 'as', 'cy', 'ru', 'sr', 'mr', 'ko', 'fa', 'ur', 'xh', 'bn', 'hr', 'pl', 'cs', 'tr', 'ne', 'it', 'ky', 'hu', 'so']]

openai_en_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]
openai_en_prompts = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        default="OpenCLIP",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--model-path",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="GPU",
    )
    parser.add_argument(
        "--open-clip", 
        type=bool, 
        default=False, 
        required=False,
        help="Indicates whether model is fine-tuned (True) or is the original OpenCLIP (False)"
    )
    parser.add_argument(
        "--adapter", 
        type=str,
        default=None, 
        required=False, 
        help="Path to adapter checkpoint"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Specified dataset.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Image batch size."
    )    
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for ImageNet dataloader."
    )  
    parser.add_argument(
        "--languages", 
        type=str, 
        default="PT",
        help="List of languages to be used for the dataset. If 'xlmr', all languages are used."
    )
    parser.add_argument(
        "--prompts", 
        type=str, 
        default="imagenet_prompts",
        help="List of prompts to be used for the dataset. If 'all', all prompts are used."
    )
    parser.add_argument(
        "--imagenet_folder", 
        type=str,
    )
    args = parser.parse_args()

    return args


# Example subclass for torchvision ImageNet dataset to load only the subset of one language
class BabelImageNet(torchvision.datasets.ImageNet):
    def __init__(self, root: str, idxs, split: str = "val", download=None, **kwargs) -> None:
        super().__init__(root, split, **kwargs)
        examples_per_class = len(self.targets) // 1000
        select_idxs = [idx*examples_per_class + i for idx in idxs for i in range(examples_per_class)]
        self.targets = [i for i in range(len(idxs)) for _ in range(examples_per_class)]
        self.imgs = [self.imgs[i] for i in select_idxs]
        self.samples = [self.samples[i] for i in select_idxs]
        self.idxs = idxs


def get_data(args, transformation, tokenizer):
    image_dataset = torchvision.datasets.ImageNet(
        args.imagenet_folder, 
        split="val", 
        loader=Image.open, 
        transform=transformation
    )

    if args.languages == "xlmr":
        languages = xlmr_langs
    else:
        languages = [l.upper() for l in args.languages.split(",")]

    text_collate = Collator(tokenizer)
    text_datasets = []

    babel_imagenet = json.load(open(os.path.join("evaluate", "utils", "resources", f"babel_imagenet.json"), "r", encoding="utf-8"))
    if "," not in args.prompts:
        prompt_names = [args.prompts]
    else:
        prompt_names = args.prompts.split(",")
    for prompt_name in prompt_names:
        lang_prompts = None
        prompts = None
        if prompt_name == "label":
            prompts = ["{}"]
        elif prompt_name == "openai_en":
            prompts = openai_en_prompts
        else:
            lang_prompts = json.load(open(os.path.join("evaluate", "utils", "resources", f"{prompt_name}.json"), "r", encoding="utf-8"))
        for lang in languages:
            try:
                if lang == "EN":
                    idxs = list(range(1000))
                    labels = openai_en_classes
                    if lang_prompts is not None:
                        prompts = openai_en_prompts
                else:
                    idxs, labels = babel_imagenet[lang]
                    if lang_prompts is not None:
                        prompts = lang_prompts[lang]
                ds = BabelImageNetTextDataset(labels=labels, prompts=prompts, class_idxs=idxs, language=lang, prompt_name=prompt_name)
                text_datasets.append(ds)
            except:
                print(f"Failed to create dataset for language {lang} with prompts {prompt_name}")
    return image_dataset, text_datasets, text_collate


class BabelImageNetTextDataset(Dataset):
    def __init__(self, labels, prompts, class_idxs, language, prompt_name):
        self.lang = language
        self.class_idxs = class_idxs
        self.prompt = prompt_name
        self.num_prompts = len(prompts)
        self.data = [prompt.format(label) for label in labels for prompt in prompts]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        text_features = self.tokenizer(examples)
        return text_features
    

def compute_accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run_evaluation(image_embeddings, text_embeddings, num_prompts=1, images_per_class=50, num_classes=1000):
    # Prompt ensembles are averaged for the final prompt embedding
    if num_prompts > 1:
        text_embeddings = text_embeddings.view(len(text_embeddings)//num_prompts, num_prompts, -1)
        text_embeddings = torch.mean(text_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)

    image_embeddings = image_embeddings.to(device=text_embeddings.device, dtype=text_embeddings.dtype)

    scores = image_embeddings @ text_embeddings.t()
    target = torch.arange(0, len(text_embeddings)).repeat_interleave(images_per_class, 0).to("cuda")

    acc1, acc5 = compute_accuracy(scores, target, topk=(1, 5))

    top1 = (acc1 / (num_classes*images_per_class))
    top5 = (acc5 / (num_classes*images_per_class))

    return top1, top5


def compute_image_embeddings(image_dataset, model, args):
    dataloader = DataLoader(image_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    all_embeddings = None
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(dataloader, desc="Encoding images"):
            batch = batch[0].to("cuda")
            embeddings = model.encode_visual(batch)
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
    return all_embeddings


def compute_text_embeddings(text_dataset, collate, model, args):
    dataloader = DataLoader(text_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    all_embeddings = None
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(dataloader, desc="Encoding text"):
            batch = batch.to("cuda")
            embeddings = model.encode_text(batch)
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
    return all_embeddings


def main():
    args = parse_args()
    print(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    if args.open_clip:
        if args.adapter is None:
            model = OpenCLIPWrapper.load_from_checkpoint(args.model_path, strict=False).model
            output_path = os.path.join(args.save_dir, args.exp_name+".txt")
            id_name = args.model_path.split("/")[-3]
        else:
            model = OpenCLIPAdapter(inference=True, devices=device)
            model.load_adapters(pretrained_adapter=args.adapter)
            output_path = os.path.join(args.save_dir, "adapter", args.exp_name+".txt")
            id_name = args.adapter
    else:
        model = OpenCLIP()
        output_path = os.path.join(args.save_dir, f"baseline_open_clip-{args.languages}.txt")
        id_name = f"baseline_open_clip_{args.languages}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vision_processor = model.image_preprocessor
    text_tokenizer = model.text_tokenizer

    model.eval()
    model.to("cuda")

    images_per_class = 50 # True for ImageNet validation split
    image_dataset, text_datasets, text_collate = get_data(args, vision_processor, text_tokenizer)

    image_embeddings = compute_image_embeddings(image_dataset, model, args)
    for text_dataset in tqdm(text_datasets, desc="Evaluating languages and prompts"):
        text_embeddings = compute_text_embeddings(text_dataset, text_collate, model, args)
        subset_image_embeddings_mask = [idx*images_per_class + i for idx in text_dataset.class_idxs for i in range(images_per_class)]
        acc1, acc5 = run_evaluation(
            image_embeddings[subset_image_embeddings_mask], 
            text_embeddings,
            num_prompts=text_dataset.num_prompts, 
            images_per_class=images_per_class,
            num_classes=len(text_dataset.class_idxs)
        )

        with open(output_path, 'a+') as file:
            
            file.write(
                f"ID: {id_name} | "\
                f"Languages: {text_dataset.lang} | "\
                f"Dataset: ImageNet1K | "\
                f"NumClasses: {len(text_dataset.class_idxs)} | "\
                f"Metric: Accuracy | "\
                f"TOP1: {100*acc1:.2f} | "\
                f"TOP5: {100*acc5:.2f}\n"\
            )


if __name__ == '__main__':
    main()