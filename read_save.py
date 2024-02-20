import json

#first we have to open our thesaurus file and iterate through it in order to create our nested dictionary. we have removed the intro and epilogue in order to make the reading easier
#save our filepath
def read():

    #first we have to open our thesaurus file and iterate through it in order to create our nested dictionary. we have removed the intro and epilogue in order to make the reading easier
    #save our filepath
    file_path = 'thesaurus.txt'

    #initialize our variables
    current_class = None
    current_division = None
    current_section = None

    #iterate over the file, while creating our nested dict
    with open(file_path, 'r') as file:
        for line in file:
            #remove leading and trailing whitespaces
            line = line.strip()

            #check if the line represents a class
            if line.startswith('CLASS'):
                current_class = line
                dictionary[current_class] = {}
                current_division = None
                current_section = None  # Reset current_section when encountering a new class

            #check if the line represents a division
            elif line.startswith('(Division'):
                current_division = line
                if current_class not in dictionary:
                    dictionary[current_class] = {}
                dictionary[current_class][current_division] = {}
                current_section = None  # Reset current_section when encountering a new division

            #check if the line represents a section
            elif line.startswith('I.') or line.startswith('II.') or line.startswith('III.') or line.startswith('IV.') or line.startswith('V.') or line.startswith('VI.') or line.startswith('VII.') or line.startswith('VIII.'):
                current_section = line
                if current_class not in dictionary:
                    dictionary[current_class] = {}
                if current_division is not None:
                    if current_division not in dictionary[current_class]:
                        dictionary[current_class][current_division] = {}
                    dictionary[current_class][current_division][current_section] = {'words': []}
                else:
                    dictionary[current_class][current_section] = {'words': []}

            #if the line contains words 
            elif current_class is not None and current_section is not None:

                #append words to the current section
                if current_division is not None:
                    dictionary[current_class][current_division][current_section]['words'].append(line)
                else:
                    dictionary[current_class][current_section]['words'].append(line)

    return None       
   
def save():
    #save out dictionary in json format
    json_file_path = 'thesaurus.json'


    with open(json_file_path, 'w') as json_file:
     json.dump(dictionary, json_file, indent=2)

print('json file saved correctly')















#create an empty dict
dictionary = {}

#read our file
read()

#and save it in json format
save()