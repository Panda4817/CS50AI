import calendar
import csv
import sys


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Read data from file
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        # Initialise tuple to return
        data = ([], [])

        # Loop over each row of data
        for row in reader:
            # Initialise evidence list
            evidence = []
            # Find month number for row 10
            month_num = 0
            for i in range(1, 13):
                if calendar.month_abbr[i] == row[10]:
                    month_num = i - 1
            # Put data into evidence list, converting numbers to int or floats
            evidence.extend([int(row[i]) if i == 0 or i % 2 ==
                             0 else float(row[i]) for i in range(len(row[:5]))])
            evidence.extend([float(cell) for cell in row[5:10]])
            evidence.append(month_num)
            evidence.extend([int(cell) for cell in row[11:15]])
            evidence.append(1 if row[15] == 'Returning_Visitor' else 0)
            evidence.append(1 if row[16] == 'TRUE' else 0)
            data[0].append(evidence)
            data[1].append(1 if row[17] == 'TRUE' else 0)
    # Return data
    return data


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Create a k-nearest neighbor model
    model = KNeighborsClassifier(n_neighbors=1)
    # Return the fitted model
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Initialise variables to calculate proportion
    total_positives = 0
    total_negatives = 0
    true_positives = 0
    true_negatives = 0
    # Loop over labels and lists
    for l, p in zip(labels, predictions):
        # Check if they the label is positive, add to variables
        if l == 1:
            total_positives += 1
            if l == p:
                true_positives += 1
        # Or the label is negative, add to variables
        else:
            total_negatives += 1
            if l == p:
                true_negatives += 1
    # Return tuple with proportions
    return (true_positives / total_positives, true_negatives / total_negatives)


if __name__ == "__main__":
    main()
