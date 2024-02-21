import json
from gensim.models import KeyedVectors
import gensim.downloader as api
import re
from unidecode import unidecode


def read_class_dictionary():
    #first we have to open our thesaurus file and iterate through it in order to create our nested dictionary. we have removed the intro and epilogue in order to make the reading easier
    #save our filepath
    classes = {}

    # Initialize variables to track current class, division, section, and title
    current_class = None
    current_division = None
    current_section = None
    count = 0

    #read the file
    with open('thesaurus.txt', 'r') as file:
        for line in file:
            #remove leading and trailing whitespaces
            line = line.strip()

            #find classes
            if line.startswith('CLASS'):
                current_class = line
                classes[current_class] = {'sections': {}}
                current_division = None
                current_section = None  # Reset current_section when encountering a new class

            #find divisions
            elif line.startswith('(Division'):
                current_division = line
                if current_class not in  classes:
                    classes[current_class] = {'sections': {}}
                classes[current_class]['sections'][current_division] = {'words': []}
                current_section = None  # Reset current_section when encountering a new division

            #find sections
            elif line.startswith('SECTION'):
                current_section = line
                if current_division is not None:
                    classes[current_class]['sections'][current_division]['sections'][current_section] = {'words': []}
                else:
                    classes[current_class]['sections'][current_section] = {'words': []}

            #else the line has words, that have to be appended
            elif current_class is not None and current_section is not None:
                
                #to the division if there is one
                if current_division is not None:
                    classes[current_class]['sections'][current_division]['sections'][current_section]['words'].append(line)
                #else, they are appended to the section
                else:
                    classes[current_class]['sections'][current_section]['words'].append(line)
    return classes 



def read_hash():
    #this function reads the dictionary to map # words to other words

    #create another dict for  hash words
    hash_dict = {}

    # Open the file and iterate through it
    with open('thesaurus.txt', 'r') as file:
        current_number = None
        current_word_list = None

        for line in file:
            # Remove leading and trailing whitespaces
            line = line.strip()

            # Check if the line starts with '#' followed by a number
            if line.startswith('#'):
                # Extract the number and create a new list for words
                current_number = line.split('.', 1)[0]
                current_word_list = [line.split('.', 1)[1].strip()]
                hash_dict[current_number] = current_word_list
            elif current_number is not None and not line.isupper():
                # Append words to the current list, ignoring uppercase lines
                current_word_list.append(line)

    return hash_dict

def save_hash_dictionary(hash_dict):
    #save our hash dictionary in json format

    json_file_path = 'hash.json'

    with open(json_file_path, 'w') as json_file:
        json.dump(hash_dict, json_file, indent= 2)

    print('json file saved correctly 2')


def save_class_dictionary(classes):

    #save out class_dictionary in json format
    json_file_path = 'classes.json'


    with open(json_file_path, 'w') as json_file:
     json.dump(classes, json_file, indent=2)

    print('json file saved correctly')


def read_embeddings(hashdict, glove_vectors):
    # Create an empty embeddings dict
    embeddings_dict = {}
    count1 = 0
    count2 = 0

    # Iterate through the list of words for each key in the hash dictionary
    for key, word_list in hashdict.items():
        # Assuming the first item in the list is the word
        word = preprocess_word(word_list[0])
        
        try:
            embeddings = glove_vectors[word]
            embeddings_dict[key] = embeddings.tolist()
            count1 += 1
        except KeyError:
            count2 += 1
            
    print(count1, count2)
    return embeddings_dict



def intialize_word2vec():
    #initialize our models

    glove_vectors = api.load('glove-twitter-25')
    return glove_vectors


def preprocess_word(word):
    # Split the string using '.' as a delimiter and take the first part
    word = word.split('.')[0]
    # Remove diacritics and convert to lowercase
    word = unidecode(word)
    # Remove special characters and convert to lowercase
    return re.sub(r'[^a-zA-Z0-9]', '', word).lower()


#START
#read our file and store it in a dictionary based on classes, divisions, and sections
classes = read_class_dictionary()

#read our file and store it in another dictionary, based on hash numbers before word categories
hash_dict = read_hash()


#and save them both in json format
save_class_dictionary(classes)
save_hash_dictionary(hash_dict)

#now we have to initialize our word2vec
glove_vectors = intialize_word2vec()

#then we read our embeddings.
embeddings = read_embeddings(hash_dict, glove_vectors)

print(repr(hash_dict['#665'][0]))
#print(embeddings['#665'])
#print(glove_vectors.most_similar(hash_dict['#665'][0]))
#print(embeddings['#1'])