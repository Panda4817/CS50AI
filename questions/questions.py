import nltk
import sys
import os
import string
import math
import operator
import itertools

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # Initialise dictionary
    files = dict()
    # Go through files in the dictionary and read it
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            files[filename] = f.read()

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Extract individual words, punctuation etc
    words = nltk.word_tokenize(document)
    # Initialise a list for filtering
    filtered = []
    # Extract words that are not punctuation and stopwords
    for w in words:
        # Lowercase word first so can be compared with stopwords list
        l = w.lower()
        if l not in string.punctuation and l not in nltk.corpus.stopwords.words("english"):
            # Append word to filtered list
            filtered.append(l)

    return filtered


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Extract all the words from all the documents and store in a set
    words = set()
    for filename in documents:
        words.update(documents[filename])
    # Initialise a dictionary
    idfs = dict()
    # Go over the words and compute idfs
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Initialise dictionary to store tf-idf values
    tfidfs = dict()
    for filename in files:
        # Calculate the tf values for query words in each document
        tfidfs[filename] = 0
        num_words = len(files[filename])
        for word in query:
            if word in files[filename]:
                tf = 0
                for w in range(len(files[filename])):
                    if files[filename][w] == word:
                        tf += 1
                # Calculate the tf-idf values
                tf_idf = tf * idfs[word]
                # Add all the tf-idf values for query words together
                tfidfs[filename] += tf_idf

    # Sort the dictionary according to highest sums
    sorted_tfidfs = sorted(
        tfidfs.items(), key=operator.itemgetter(1), reverse=True)
    # Return n number of files ranked according to tf-idf sums
    return [pair[0] for pair in itertools.islice(sorted_tfidfs, n)]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Initialise dictionary to store calculated values
    qtd_idfs = dict()
    for s in sentences:
        # Initialise variables to calculate sum of idfs and query term density
        sum_idfs = 0
        matching_words = 0
        sentence_words = len(sentences[s])
        # Loop over words in the query and check if the sentence has them
        for word in query:
            if word in sentences[s]:
                # Calculate sum of idfs
                sum_idfs += idfs[word]
                # Keep track of the number of matching words
                matching_words += sentences[s].count(word)
        # Calculate query term density
        qtd = matching_words / sentence_words
        qtd_idfs[s] = (sum_idfs, qtd)

    # Sort sentences by idfs and then query term density
    sorted_list = sorted(qtd_idfs.items(), key=lambda k: (k[1][0], k[1][1]))
    sorted_list.reverse()
    return [sentence[0] for sentence in itertools.islice(sorted_list, n)]


if __name__ == "__main__":
    main()
