import csv
import math
import random
import operator
import logging

logging.basicConfig(format='[%(asctime)s] [%(name)s:%(lineno)d] | [%(levelname)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filename, split, trainingSet=None, testSet=None):
    """load IRIS dataset and randomly split it into test set and training set."""
    if trainingSet is None:
        trainingSet = []
    if testSet is None:
        testSet = []
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        next(lines, None)  # skip the headers
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclidean_distance(instance1, instance2, length):
    """euclidean distance calculation."""
    distance = 0
    for x in range(length-1):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(trainingSet, testInstance, k):
    """selecting subset with the smallest distance."""
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclidean_distance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_predicted_response(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def get_accuracy(testSet, predictions):
    """Calculate accuracy."""
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] in predictions[x]:
            correct = correct + 1

    return correct / float(len(testSet)) * 100


def apply_knn():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67  # 67% of input dataset is for training, rest is used for test-dataset
    logger.info("--- Begin Loading DataSet --- ")

    load_data('./Iris.csv', split, trainingSet, testSet)

    logger.info('Training-Set SIZE: ' + repr(len(trainingSet)))
    logger.info('Test-Set SIZE: ' + repr(len(testSet)))
    logger.info("--- Loading DataSet Completed.--- ")

    # generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = get_neighbors(trainingSet, testSet[x], k)
        result = get_predicted_response(neighbors)
        predictions.append(result)
        logger.info('predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = get_accuracy(testSet, predictions)
    logger.info('Accuracy: ' + repr(accuracy) + '%')


if __name__ == "__main__":
    """Start KNN Process."""
    apply_knn()
