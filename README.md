﻿# Roget-s-Thesaurus-Data-Analysis-Project
 Roget's Thesaurus is a dictionary that was written in 1852. Classes contain, or may not contain, divisions, which then contain subclasses which then contain titles that contain words.
 For each set of words, we get its modern day embedding from word2vec(an embeddings api), trained by a twitter model. Then, based on those embeddings, we proceed to cluster the words into classes and then subclasses.
 That is done, in order to get a grasp of the notion pertaining to each word's modern day meaning, and then proceed to compare it with how the notion of the word was, 200 years ago, when Roget wrote his Thesaurus.
You only need the thesaurus.txt, main.py, rogets_thesaurus.ipynb and pyproject.toml files to run the scrip(and of course the README file is useful as well).
When executing the rogets_thesaurus.ipynb file, main.py will execute, and the rest of the (json)files will be generated.
The modern_dictionary_with_names.json file contains our new cluster classes and their cluster sections(the clustering based on their average embeddings), along with the words and names for each new class and section.
The rest of the dictionaries were helpful whilst writing the code.
