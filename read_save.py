import json

#first we have to open our thesaurus file and iterate through it in order to create our nested dictionary. we have removed the intro and epilogue in order to make the reading easier
#save our filepath
def read_dictionary():

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
                dictionary[current_class] = {'sections': {}}
                current_division = None
                current_section = None  # Reset current_section when encountering a new class

            #find divisions
            elif line.startswith('(Division'):
                current_division = line
                if current_class not in dictionary:
                    dictionary[current_class] = {'sections': {}}
                dictionary[current_class]['sections'][current_division] = {'words': []}
                current_section = None  # Reset current_section when encountering a new division

            #find sections
            elif line.startswith('SECTION'):
                current_section = line
                if current_division is not None:
                    dictionary[current_class]['sections'][current_division]['sections'][current_section] = {'words': []}
                else:
                    dictionary[current_class]['sections'][current_section] = {'words': []}

            #else the line has words, that have to be appended
            elif current_class is not None and current_section is not None:
                
                #to the division if there is one
                if current_division is not None:
                    dictionary[current_class]['sections'][current_division]['sections'][current_section]['words'].append(line)
                    count += 1
                #else, they are appended to the section
                else:
                    dictionary[current_class]['sections'][current_section]['words'].append(line)
                    count +=1
        print(count)
        




def save_dictionary():
    #save out dictionary in json format
    json_file_path = 'thesaurus.json'


    with open(json_file_path, 'w') as json_file:
     json.dump(dictionary, json_file, indent=2)

    print('json file saved correctly')




#create an empty dict
dictionary = {}

#read our file
read_dictionary()

#and save it in json format
save_dictionary()
