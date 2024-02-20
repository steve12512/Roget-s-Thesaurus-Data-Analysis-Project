import json


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

    # Print the resulting dictionary
    for key, value in hash_dict.items():
        print(hash_dict.keys)

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

#START
#read our file and store it in a dictionary based on classes, divisions, and sections
classes = read_class_dictionary()

#read our file and store it in another dictionary, based on hash numbers before word categories
hash_dict = read_hash()



#and save them both in json format
save_class_dictionary(classes)
save_hash_dictionary(hash_dict)


print(hash_dict['#1'])