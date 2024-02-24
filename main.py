import json
from gensim.models import KeyedVectors
import gensim.downloader as api
import re
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def read_class_dictionary():
    #first we have to open our thesaurus file and iterate through it in order to create our nested dictionary. we have removed the intro and epilogue in order to make the reading easier
    #create empty dictionary
    classes = {}

    #initialize variables to track current class, division, section, and title
    current_class = None
    current_division = None
    current_section = None


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
                classes[current_class]['sections'][current_division] = {'words': [], 'numbers': []}
                current_section = None  # Reset current_section when encountering a new division

            #find sections
            elif line.startswith('SECTION'):
                current_section = line
                if current_division is not None:
                    classes[current_class]['sections'][current_division]['sections'][current_section] = {'words': [], 'numbers': []}
                else:
                    classes[current_class]['sections'][current_section] = {'words': [], 'numbers': []}

            #else the line has words, that have to be appended
            elif current_class is not None and current_section is not None:
                
                #to the division if there is one
                if current_division is not None:

                    if line.startswith('#'):
                        #if the line starts with '#' followed by a number, extract and append the number

                        numbers = re.findall(r'#(\d+)', line)
                        classes[current_class]['sections'][current_division]['sections'][current_section]['numbers'].extend(numbers)

                    #otherwise, it is a word
                    else:
                        classes[current_class]['sections'][current_division]['sections'][current_section]['words'].append(line)
                
                #else, they are appended to the section in the same way
                else:

                    if line.startswith('#'):
                        
                        numbers = re.findall(r'#(\d+)', line)
                        classes[current_class]['sections'][current_section]['numbers'].extend(numbers)
                    
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



def save_class_dictionary(classes):

    #save out class_dictionary in json format
    json_file_path = 'classes.json'


    with open(json_file_path, 'w') as json_file:
     json.dump(classes, json_file, indent=2)


def read_embeddings(hashdict, glove_vectors):
    # Create an empty embeddings dict
    embeddings_dict = {}
    count1 = 0
    count2 = 0

    # Iterate through the list of words for each key in the hash dictionary
    for key, word_list in hashdict.items():
        # Iterate through all words in the list
        for word_entry in word_list:
            # Extract individual words
            words = word_entry.split('.')
            for word in words:
                # Preprocess each word
                word = preprocess_word(word.strip())
                
                try:
                    embeddings = glove_vectors[word]
                    embeddings_dict[key] = embeddings.tolist()
                    count1 += 1
                    # Break after finding embeddings for one word in the list
                    break
                except KeyError:
                    count2 += 1
    return embeddings_dict


def save_embedings_dictionary(embeddings):
    #save our hash dictionary in json format

    json_file_path = 'embeddings.json'

    with open(json_file_path, 'w') as json_file:
        json.dump(embeddings, json_file, indent= 2)


def save_average_embeddings_dictionary():
    # Convert keys to strings and handle non-serializable values
    converted_embeddings = {}
    for key, value in average_embeddings.items():
        converted_value = value.copy()  # Create a copy to avoid modifying the original dictionary
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, np.int32):
                converted_value[sub_key] = int(sub_value)  # Convert numpy int32 to Python int
        converted_embeddings[str(key)] = converted_value

    # Save the dictionary in JSON format
    json_file_path = 'average_embeddings.json'

    with open(json_file_path, 'w') as json_file:
        json.dump(converted_embeddings, json_file, indent=2)

def intialize_word2vec():
    #initialize our models

    glove_vectors = api.load('glove-twitter-25')
    return glove_vectors


def preprocess_word(word):
    # Split the string using '.' as a delimiter and take the first part word = word.split('.')[0]
    # Remove diacritics and convert to lowercase
    #word = unidecode(word)
    # Remove special characters and convert to lowercase
    return re.sub(r'[^a-zA-Z0-9]', '', word).lower() #START



def get_average_embeddings(embeddings):
    #this function is used to create a dictionary of embeddings, only this time we keep the average within the list of embeddings from our previous embeddings dictionary. that is, so that we can create clusters more easily

    #first create an empty dictionary as usual
    average_embeddings = {}

    for key, values in embeddings.items():

        number = 0
        for value in values:
            number += value
        number = number / len(values)
        average_embeddings[key] = number
    return average_embeddings





def average_class_clusters(average_embeddings, num_clusters=5):
    # Assuming you have your average_embeddings dictionary
    data = list(average_embeddings.values())

    # Reshape the data to make it 2D using numpy
    data_reshaped = np.array(data).reshape(-1, 1)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)  # Explicitly set n_init
    clusters = kmeans.fit_predict(data_reshaped)

    # Add cluster labels to the average_embeddings dictionary
    for i, (key, value) in enumerate(average_embeddings.items()):
        average_embeddings[key] = {'value': value, 'cluster': clusters[i]}

    # Print cluster statistics
    for cluster in range(num_clusters):
        cluster_data = [value['value'] for value in average_embeddings.values() if value['cluster'] == cluster]
        print(f"Cluster {cluster + 1} - Number of elements: {len(cluster_data)}")
        print(f"       Mean: {np.mean(cluster_data)}")
        print(f"       Std Dev: {np.std(cluster_data)}")

    # Visualize clusters
    plt.scatter(range(len(data)), data, c=clusters, cmap='viridis')
    plt.title('K-Means Clustering')
    plt.xlabel('Data Point Index')
    plt.ylabel('Value')
    plt.show()



def modify_average_embeddings():
    #modify our dictionary, so that it also contains its original class number
    for class_name, class_data in classes.items():
        for section_name, section_data in class_data['sections'].items():
                for number in section_data['numbers']:
                    number = "#" + str(number)
                    average_embeddings[number] = { 'value': average_embeddings[number]['value'], 'cluster': average_embeddings[number]['cluster'], 'original class' : class_name, 'word' : str(hash_dict[number])[1:12]}
                        #average_embeddings[key] = {'value': value, 'cluster': clusters[i]}
                    



def create_clusters_dictionary():
    #this method will save our new clusters, along with their average word embeddings, their hash numbers, and their original meaning
    clusters_dictionary = {}

    for key, value in average_embeddings.items():
        cluster_number = value.get('cluster', None)

        if cluster_number is not None:
            if cluster_number not in clusters_dictionary:
                clusters_dictionary[cluster_number] = {}

            clusters_dictionary[cluster_number][key] = {
                'value': value['value'],
                'original class': value.get('original class', None), 'word' : str(hash_dict[key])[1:12]
            }

    # Print the clusters_dictionary
    for cluster_number, cluster_data in clusters_dictionary.items():
        #print(f"Cluster {cluster_number}:")
        for key, data in cluster_data.items():

            #print(f"    {key}: {data}") #print()
            pass
    return clusters_dictionary



def save_clusters_dictionary(json_file_path='clusters_dictionary.json'):
    # Convert keys to strings
    clusters_dict_str_keys = {str(key): value for key, value in clusters_dictionary.items()}

    # Save the clusters dictionary in JSON format
    with open(json_file_path, 'w') as json_file:
        json.dump(clusters_dict_str_keys, json_file, indent=2)



def perform_section_clustering(cluster_data, class_name, num_clusters=5):
    # This function will generate new section clusters for a class.
    # It will then return the mean of these embeddings to the sections dictionary, as a list.
    mean_embeddings = []

    print(f"\nClass: {class_name}")

    # Extract relevant information from cluster_data
    data = [(key, value['value'], value['original class'], value['word']) for key, value in cluster_data.items()]
    hash_numbers, embeddings, original_classes, words = zip(*data)

    # Reshape the data to make it 2D using numpy
    data_reshaped = np.array(embeddings).reshape(-1, 1)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_reshaped)

    # Print cluster statistics and create modern_dictionary structure
    modern_dictionary = {}
    for cluster in range(num_clusters):
        cluster_indices = [i for i in range(len(data)) if clusters[i] == cluster]

        print(f"Cluster {cluster + 1} - Number of elements: {len(cluster_indices)}")
        print(f"       Mean: {np.mean([embeddings[i] for i in cluster_indices])}")
        print(f"       Std Dev: {np.std([embeddings[i] for i in cluster_indices])}")

        cluster_entries = {hash_numbers[i]: {
            'value': embeddings[i],
            'original class': original_classes[i],
            'word': words[i]
        } for i in cluster_indices}

        modern_dictionary[f"Section {cluster + 1}"] = cluster_entries

    # Visualize clusters
    plt.scatter(range(len(embeddings)), embeddings, c=clusters, cmap='viridis')
    plt.title(f'K-Means Clustering - Class: {class_name}')
    plt.xlabel('Data Point Index')
    plt.ylabel('Value')
    plt.show()

    return modern_dictionary


def get_section_clusters():  
    # For each cluster, perform a clustering to get new cluster sections,
    # and do that by finding their cluster sections embeddings.
    # Create a new dictionary to contain the new section cluster centers for each cluster(class)

    # Create a new dictionary that will map each cluster name to its new cluster sections. This will be our final dictionary.
    modern_dictionary = {}
    for cluster_name, cluster_data in clusters_dictionary.items():
        # Perform k-means clustering for the cluster and append the new cluster centers(sections) to our new dictionary
        modern_dictionary[cluster_name] = perform_section_clustering(cluster_data, cluster_name)

    return modern_dictionary






def save_modern_dictionary():
    with open('modern_dictionary.json', 'w') as json_file:
        # Convert keys to strings using a new dictionary
        string_keys_dictionary = {str(key): value for key, value in modern_dictionary.items()}
        json.dump(string_keys_dictionary, json_file, indent=2)



def rename_modern_dictionary():
    class_names = [
        "CLASS I. Philosophical Taxonomy",
        "CLASS II. Diverse Manifestations",
        "CLASS III. Life Dynamics",
        "CLASS IV. Transformational Entities",
        "CLASS V. Elemental Forces",
        "CLASS VI. Societal Dynamics"
    ]

    section_names = [
        "Existence and Essence",
        "Relations and Dispositions",
        "Connection and Manifestation",
        "Differentiation and Manifested Properties",
        "Comparison and Qualification",
        "Exclusion and Novelty",
        "Extrinsical Identity and Forms",
        "Decomposition and Diversity",
        "Similarity and Compositions",
        "Junctions and Manifestations",
        "Middle and Motives",
        "Uniformity and Ends",
        "State and Events",
        "Absolute Properties and Entities",
        "Junctions and Intentions",
        "Transformation and Contractions",
        "Imitation and Variations",
        "Disagreements and Destruction",
        "Inferiority and Sound",
        "Combination and Possession",
        "Error and Misfortune",
        "Absence and Odors",
        "Smallness and Memory",
        "Bitterness and Indications",
        "Assignments and Demonstration",
        "Youth and Expressive Forces",
        "Support, Direction, and Social Influences",
        "Size, Clothing, and Relationship Dynamics",
        "Change, Evening, and Governance Forces",
        "Demonstration and Financial Forces"
    ]
    
    # Create a new dictionary with class names as keys
    modern_dictionary_with_class_names = dict(zip(class_names, modern_dictionary.values()))

    # Update section names within each class
    for class_name, sections in modern_dictionary_with_class_names.items():
        updated_sections = {}
        for section_name, inner_dict in sections.items():
            section_number = int(section_name.split()[-1])
            section_name_updated = section_names[section_number - 1]
            updated_sections[section_name_updated] = inner_dict
        modern_dictionary_with_class_names[class_name] = updated_sections

    # Save the modified dictionary to a JSON file
    output_file_path = 'modern_dictionary_with_names.json'
    with open(output_file_path, 'w') as json_file:
        json.dump(modern_dictionary_with_class_names, json_file, indent=2)

    return modern_dictionary_with_class_names






#START OF OUR PROGRAM



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



#get the average embeddings for each list in the #numbers
average_embeddings = get_average_embeddings(embeddings)


#save_average_embeddings_dictionary(average_embeddings)
save_embedings_dictionary(embeddings)



#generate class clusters
average_class_clusters(average_embeddings, 6)


#add the name of the original class to our new cluster classes
modify_average_embeddings()





#save it in a dictionary
save_average_embeddings_dictionary()


#create a dictionary that stores our new class clusters and the embeddings of the words within them, their hash number, and their original meaning
clusters_dictionary = create_clusters_dictionary()

#then save that in a json file as well
save_clusters_dictionary()





#this will generate new section clusters, for each class cluster. it will be our final dictionary
modern_dictionary = get_section_clusters()



save_modern_dictionary()

#give our class clusters names
modern_dictionary_with_class_names = rename_modern_dictionary()
