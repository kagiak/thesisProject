import os
import sys
import json
import copy
import csv

# the layer-merge code is provided by the recipe1m team
def dspath(ext, ROOT, **kwargs):
    return os.path.join(ROOT, ext)

class Layer(object):
    L1 = 'recipe1M/layer1'
    L2 = 'recipe1M/layer2'
    L3 = 'recipe1M/layer3'
    INGRS = 'recipe1M/det_ingrs'
    GOODIES = 'goodies'

    @staticmethod
    def load(name, ROOT, **kwargs):
        with open(dspath(name + '.json', ROOT, **kwargs)) as f_layer:
            return json.load(f_layer)

    @staticmethod
    def merge(layers, ROOT, copy_base=False, **kwargs):
        layers = [l if isinstance(l, list) else Layer.load(l, ROOT, **kwargs) for l in layers]
        base = copy.deepcopy(layers[0]) if copy_base else layers[0]
        entries_by_id = {entry['id']: entry for entry in base}
        for layer in layers[1:]:
            for entry in layer:
                base_entry = entries_by_id.get(entry['id'])
                if not base_entry:
                    continue
                base_entry.update(entry)
        return base

def getIngredient(key, i):
    """
        getIngredients() is a function to retrieve the ingredients
        according to my ingredient database ingredient.csv from
        the one with the raw ingredients for each image.
        If one of the 602 ingredients is found inside the raw ingredients
        of the recipe, as retrieved from the web, it adds it to the list
        of ingredients for the specific image.

        Key is the element from my ingredient list ingredients.csv.
        Though i is the raw ingredient taken from the layer.json files
        which are retrieved from the websites.

        This function checks if the keys exists inside the raw information.
        I.e. key=tomato and i=small tomatoes then the key is added to the
        ingredients list for this image. Because it checks if tomato or its
        plural tomatoes exist in the raw info.

    """
    key = key.replace("_"," ")
    if (key in i):
        ingr = key.split(" ")
        ingr1 = i.split(" ")
        length = len(ingr)
        length1 = len(ingr1)
        if ((key == i) or (''.join((key, 's')) == i) or (''.join((key, 'es')) == i)): # in case that is only one word the ingredient
            if key not in listOfIngr:
                listOfIngr.append(key)
        elif ((length == 1) & (length1 > 1)): # in case the ingredient from the website has more than one words
            for z in ingr1:
                if ((z == key) or (''.join((key, 's')) == z)or (''.join((key, 'es')) == z)):
                    if key not in listOfIngr:
                        listOfIngr.append(key)
        elif((length > 1) & (length1 == 1)):
            for n in ingr:
                if ((n == key) or (''.join((key, 's')) == n)or (''.join((key, 'es')) == n)):
                    if key not in listOfIngr:
                        listOfIngr.append(key)
        elif ((length1 > 1) & (length > 1)): #in case both ingredients are more than one words
            l = 0
            for x in ingr1:
                for y in ingr:
                    if ((x == y) or (''.join((x, 's')) == y)or (''.join((key, 'es')) == y)): # check if any words from the 2 strings are equal
                        l = l + 1
            if (l == length):
                if key not in listOfIngr:
                    # print (key)
                    listOfIngr.append(key)

if __name__ == "__main__":
    listOfRecipes = []
    recipeIngr = []
    recipeIngredDict = {}
    imageIngredDict = {}
    errorCount = 0
    listOfIDs = []
    mylist = []

    data_path = os.path.join(os.path.dirname(sys.argv[0]))
    dataset = Layer.merge([Layer.L1, Layer.L2, Layer.INGRS], data_path) # the dataset is list, where each entry is a recipe. the recipe contains a dictionary

    with open('recipe1M/ingredients-reduc.csv', mode='r') as infile:
        reader = csv.reader(infile)
        ingrsSet = {rows[0] for rows in reader}

    # create the list with all the ingredients from the .csv file and sort it. Also create a same-size binary list
    for i in ingrsSet:
        i = i.replace("_", " ")
        mylist.append(i)
    mylist.sort()

    Ingredients = {}
    myDict = {}
    number = 0

    # start the loop for each entry in the original dataset
    for d in dataset:
        partition = d['partition']
        if partition == "test": # CHANGE partition to "train", "test", "val" to retrieve the different sets
            valid = d['valid']
            id = d['id']
            ingredients = d['ingredients']
            for number,t in enumerate(ingredients):
                if valid[number]:
                    ing = t['text']
                    recipeIngr.append(ing) # a list with the ingredients taken from the json file. Just the ingredients without the 'text' word
            listOfIngr = []
            # for every ingredient (key) in my dataset ingredients.csv (with the final 602 ingredients)
            for key in ingrsSet:
                for i in recipeIngr: # and for every i (ing taken from the json file)
                    getIngredient(key, i) # check if the key exists in the i (ing)

            if len(listOfIngr) > 1: # if we have keys then
                recipeIngredDict[ d['id']] = listOfIngr # create a dictionary [id:(listOfIngr)]
                if 'images' in d: # if the recipe has images
                    image = d['images']
                    if len(image) >= 1:
                        for i in image:
                            imageUrl = i['url']
                            imageId = i['id']
                            binarymylist = []
                            finalDict = {}
                            for i in ingrsSet:
                                binarymylist.append(0)
                            for l in listOfIngr:
                                for number,ingr in enumerate(mylist):
                                    if l == ingr:
                                        binarymylist[number] = 1

                            finalDict[imageId] = [listOfIngr, binarymylist]
                            # this way it appends dictionaries, which afterwards could not be loaded from json
                            # in my model so I replaced }{ in notepad with , and I am using this new json for training
                            with open('recipe1M/datatest2.json', mode='a+') as f: # CHANGE json's name
                                json.dump(finalDict, f)

                            del binarymylist[:]
                            finalDict.clear()
            else:
                continue
            del recipeIngr[:]
