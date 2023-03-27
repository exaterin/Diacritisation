import re
import argparse
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--train", default="model.pickle", type=str, help="Training mode: True or path to the model")
parser.add_argument("--data", default="vesmir_articles37.txt", type=str, help="Path to training data")
parser.add_argument("--test_data_no_diacritics", default="diacritics-dtest.txt_no_diacritics.txt", type=str, help="Path to test data without diacritics")
parser.add_argument("--test_data_diacritics", default="diacritics-dtest.txt", type=str, help="No or path to test data with diacritics")
parser.add_argument("--dict", default="dict.txt", type=str, help="No or path to dictionary")
parser.add_argument("--output", default="diacritized.txt", type=str, help="Path to output file")

# Translation between diacritic and non-diacritic letters
DIACR_TO_NODIACR = str.maketrans("áčďéěíňóřšťúůýž" + "áčďéěíňóřšťúůýž".upper(), "acdeeinorstuuyz" + "acdeeinorstuuyz".upper())

DIACRITISABLE = '[acdeinorstuyzáčďéěíňóřšťúůýž]'
NODIACR = '[acdeinorstuyz]'

# Create target values for given letters
def create_target(letter):
    if letter in "acdeinorstuyz":
        return 0
    if letter in "áéíóúý":
        return 1
    if letter in "čěňřšžďťů":
        return 2
    return -1

# Convert target values to letters
def target_to_letter(letter, target):
    if target == 0:
        return letter
    if target == 1:
        ind = "aeiouy".find(letter)
        if ind != -1: return "áéíóúý"[ind]
    if target == 2:
        ind = "cenrszdtu".find(letter)
        if ind != -1: return "čěňřšžďťů"[ind]
    return letter

# Create feature vectors for diacritic letters
def create_feature_vectors(data, create_targets=False):

    # Define regular expression pattern for finding diacritic letters
    candidates_pattern = re.compile(NODIACR)

    # Read in data
    with open(data, 'r', encoding='utf-8') as f:
        training_d = f.read().lower()

    # Remove diacritics from the training data
    training_data = training_d.translate(DIACR_TO_NODIACR)

    feature_vectors = []
    targets = []
    indeces = []

    # Go through the text and create feature vectors for diacritic letters
    for i, char in enumerate(training_data):

        # Check if the current character is a diacritic letter
        match = candidates_pattern.match(char)

        if match:
            # Original letter
            letter = training_d[i]

            feature_vector = []

            # Add 5 characters before the current character to the feature vector
            start = max(i - 5, 0)
            end = i
            if end - start < 5:
                feature_vector += [' '] * (5 - (end - start))
            feature_vector += list(training_data[start:end])

            # Add current character to the feature vector
            feature_vector += [training_data[i]]

            # Add 5 characters after the current character to the feature vector
            start = i + 1
            end = min(i + 6, len(training_data))
            feature_vector += list(training_data[start:end])
            if len(feature_vector) < 11:
                feature_vector += [' '] * (11 - len(feature_vector))
            

            feature_vectors.append(feature_vector)
            indeces.append(i)

            # print(feature_vector, letter, create_target(letter))

            if create_targets:
                targets.append(create_target(letter))

    return feature_vectors, indeces, targets

# Train the model
def train_model(training_data_features, targets):

    # Encode feature vectors using one-hot encoding
    print("One hot encoding...")
    onehot = OneHotEncoder(handle_unknown='ignore')
    onehot.fit(training_data_features)

    feature_vector_encoded = onehot.transform(training_data_features).toarray()

    # Train the MLP classifierS
    print("Training...")
    mlp = MLPClassifier()
    mlp.fit(feature_vector_encoded, targets)

    # Save the trained MLP classifier and one-hot encoder
    with open('model.pickle', 'wb') as f:
        pickle.dump((mlp, onehot), f)

    return mlp, onehot

# Remove diacritics from the text, use if you need to prepare test data
def remove_diacritics_and_save(data):
    
    with open(data, 'r', encoding='utf-8') as f:
        text = f.read()

        # Save text without diacritics to file
        with open(f'{data}_no_diacritics.txt', 'w', encoding='utf-8') as f:
            f.write(text.translate(DIACR_TO_NODIACR))

# Read dictionary
def read_dict(dict_path):
    dictionary = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split("\n")
        for line in lines:
            line = line.strip()
            word, variants = line.split(':')
            dictionary[word] = variants.strip().split(', ')
    return dictionary


if __name__ == '__main__':

    args = parser.parse_args()

    if args.train == "True":
        print("Creating feature vectors...")
        features, indeces, targets = create_feature_vectors(args.data, create_targets=True)
        mlp, onehot = train_model(features, targets)
        print("Done")
    else:
        # Load the saved MLP classifier and one-hot encoder
        with open(args.train, 'rb') as f:
            mlp, onehot = pickle.load(f)

    # Read test data
    with open(args.test_data_no_diacritics, 'r', encoding='utf-8') as f:
        test_data_original = f.read()


    # Use if you want to remove diacritics from the data
    # test_data = remove_diacritics_and_save(args.test_data)

    test_features, indeces, _ = create_feature_vectors(args.test_data_diacritics, create_targets=False)
    test_features_encoded = onehot.transform(test_features).toarray()

    # Predict diacritic letters
    if args.dict == "No":
        test_data_original = list(test_data_original)
        predictions = mlp.predict(test_features_encoded)

        # Replace diacritic letters in the original text
        for i, target in zip(indeces, predictions):
            if (test_data_original[i].isupper()):
                test_data_original[i] = target_to_letter(test_data_original[i].lower(), target).upper()
            else:
                test_data_original[i] = target_to_letter(test_data_original[i], target)

        # Save diacritized text to file
        with open(args.output, 'w', encoding='utf-8') as f:
            diacritized = f.write(''.join(test_data_original))

        # Calculate accuracy
        with open(args.test_data_diacritics, 'r', encoding='utf-8') as f:
            orig_data = f.read()

        accuracy = accuracy_score(list(orig_data), list(''.join(test_data_original)))

        print(f"Per-character accuracy: {accuracy * 100}%")

    else:
        dictionary = read_dict(args.dict)

        probs = mlp.predict_proba(test_features_encoded)
        pred = mlp.predict(test_features_encoded)

        words = test_data_original.lower().split()
        test_data_original = list(test_data_original)

        ind = 0
        for i, word in enumerate(words):

            count = re.findall(NODIACR, word)

            # If there are no letters to diacritize, skip the word
            if len(count) == 0:
                ind += len(count)
                continue
            
            # If word is in dictionary
            elif word in dictionary:
                variants = dictionary[word]

                # If there is only one variant, choose it
                if len(variants) == 1:
                    choosen = variants[0]
                else:
                    # Calculate sum of logarithms of probabilities of each letter for each variant
                    sum_probs = []
                    for variant in variants:
                        candidates = re.findall(DIACRITISABLE, variant)
                        sum_prob = sum(np.log(probs[ind + i][create_target(c)]) for i, c in enumerate(candidates))
                        sum_probs.append(sum_prob)

                    choosen = variants[np.argmax(sum_probs)]

                # Replace letters in original text
                for j in range(len(count)):
                    if (test_data_original[indeces[ind + j]].isupper()):
                        test_data_original[indeces[ind + j]] = re.findall(DIACRITISABLE, choosen)[j].upper()
                    else:
                        test_data_original[indeces[ind + j]] = re.findall(DIACRITISABLE, choosen)[j]
    
            # If word is not in dictionary, use the most probable variant
            else:
                count = re.findall(NODIACR, word)
                for j in range(len(count)):
                    if (test_data_original[indeces[ind + j]].isupper()):
                        test_data_original[indeces[ind + j]] = target_to_letter(test_data_original[indeces[ind + j]].lower(), pred[ind + j]).upper()
                    else:
                        test_data_original[indeces[ind + j]] = target_to_letter(test_data_original[indeces[ind + j]], pred[ind + j])

            ind += len(count)

        # Save diacritized text to file
        with open(args.output, 'w', encoding='utf-8') as f:
            diacritized = f.write(''.join(test_data_original))

        # Calculate accuracy
        with open("diacritics-dtest.txt", 'r', encoding='utf-8') as f:
            orig_data = f.read()

        accuracy = accuracy_score(list(orig_data), list(''.join(test_data_original)))

        print(f"Per-character accuracy: {accuracy * 100}%")
