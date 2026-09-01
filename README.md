# Smart-Tech-CA2

# Self-Driving Car – Smart Technologies CA2

## Overview

This project was developed for the Smart Technologies CA2 assignment.

The objective was to develop a convolutional neural network (CNN) capable of predicting steering angles from images captured using the Udacity Self-Driving Car Simulator.

Training data was generated manually using the simulator. The project includes data exploration, preprocessing, augmentation, CNN development, model evaluation and autonomous simulator testing.

Several model and dataset iterations were completed during development. The final dataset contains driving examples from both simulator tracks.

## Git Repository

GitHub Repository: https://github.com/eljesam/Smart-Tech-CA2.git

## Dataset

Training data was generated manually using the Udacity Self-Driving Car Simulator.

Each driving record contains:

Centre camera image
Left camera image
Right camera image
Steering angle
Throttle
Brake
Speed

Three images are therefore generated for each driving record.

### Initial Dataset

The initial Track 1 dataset contained:

4,361 driving records
13,083 images

The initial data contained a large proportion of straight-driving examples, which created an imbalance in the steering distribution.

Additional targeted data was collected during later experiments to improve cornering and recovery behaviour.

## Track 2 Dataset

The original Track 1 model did not generalise successfully to Track 2.

Additional Track 2 training data was therefore collected.

Track 2 dataset:

1,390 driving records
4,170 images

## Overfitting Control

Several techniques were used to reduce overfitting:

Training data augmentation
Dropout
Train/validation split
Early stopping
Model checkpointing

Training stops when validation performance no longer improves.

It was also observed that validation loss could be lower than training loss. This is reasonable because training images are augmented and dropout is active during training, while validation images are not augmented.

## Simulator Testing
### Track 1

The final combined model successfully drives through a significant portion of Track 1.

It negotiates the earlier sections of the track and reaches the approach to the long straight section before eventually leaving the track.

A complete autonomous lap was not achieved.

### Track 2

The final model was also tested autonomously on Track 2.

Although Track 2 training data was included in the final dataset, the vehicle still collided with the track boundary early in the autonomous test.

A complete Track 2 lap was not achieved.

These results show that the model learned meaningful autonomous steering behaviour but still has limitations in generalisation and recovery.

## Known Limitations

The current model does not complete a full autonomous lap on either simulator track.

Potential future improvements include:

Additional targeted Track 2 training data
More recovery examples
Better balancing of difficult driving situations
Additional hyperparameter tuning
Alternative CNN architectures
Improved throttle and speed control
More systematic validation of simulator behaviour
Additional training runs with controlled experiments

# Installation

1. Create vitual environment
python -m venv .venv
2. Activate it on Windows
.venv\Scripts\activate
3. Install required packages defined in "requirements.txt"
pip install -r requirements.txt

# Training Model
- The final training dataset should be available at:

"data/raw/driving_log_final_combined.csv"

- Training images should be available inside:

"data/raw/IMG/"

- Run training from the project root:

python src/train.py

- The final trained model is saved to:

"models/self_driving_model_final.keras"

- The training-loss graph is saved to:

"models/training_loss_final.png"

# Running Autonomous Mode

- The simulator communicates with: *drive.py*

- The default autonomous settings are:

Target speed: 8
Throttle: 0.15
Steering gain: 1.0

- Start the Python server:

python drive.py

Then:

1. Open the Udacity Self-Driving Car Simulator.
2. Select the required track.
3. Select Autonomous Mode.
4. The simulator connects to the Python server on port 4567.
5. Camera images are sent to the CNN.
6. The CNN predicts a steering angle.
7. *drive.py* sends the steering and throttle values back to the simulator.