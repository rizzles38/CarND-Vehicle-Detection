from utils import load_training, prep_training_data, train_classifer, save_classifier
import pickle

# Load the training data
cars, not_cars = load_training()
print("Found {} cars.".format(len(cars)))
print("Found {} not cars.".format(len(not_cars)))

# Prep the training data
print("Prepping data for training...")
X_train, X_test, y_train, y_test, normalizer = prep_training_data(cars, not_cars)
print("Each image has {} features.".format(X_train.shape[1]))

# Train the classifier
print("Training the classifier...")
classifier, accuracy = train_classifer(X_train, y_train, X_test, y_test)
print("Classifier accuracy = {}".format(accuracy))

# Save the classifer
save_classifier(classifier, normalizer)
print("Saved to classifier.pickle.")
