# Model Training and Testing Runs

## Run 1 - Baseline Model

### Dataset
- Original driving records: 4,361
- Balanced samples: 1,649
- Training samples: 1,319
- Validation samples: 330
- Train/validation split: 80/20

### Model
- Architecture: NVIDIA-inspired CNN
- Input size: 66 x 200 x 3
- Convolutional layers: 5
- Dropout: 0.5
- Output: Single steering-angle value

### Preprocessing
- Image cropping
- Resize to 200 x 66
- RGB to YUV conversion
- Pixel normalization inside the CNN

### Augmentation
- Random centre/left/right camera selection
- Side-camera steering correction: 0.2
- Horizontal flipping
- Random brightness adjustment

### Training Hyperparameters
- Optimizer: Adam
- Learning rate: 0.0001
- Loss: Mean Squared Error
- Batch size: 32
- Maximum epochs: 20
- Early stopping patience: 3

### Training Results
- Epochs completed: 9
- Best epoch: 6
- Best validation loss: 0.1697

### Overfitting / Underfitting
Training loss decreased substantially during the first few epochs.
Validation loss reached its lowest value at epoch 6 and then began to
increase. Early stopping prevented further unnecessary training and restored
the best model weights.

### Simulator Test - Track 1

Target speed: 12
Throttle: 0.20

Result:
Failed to complete Track 1.

Observations:
- The vehicle moved successfully in Autonomous Mode.
- The vehicle initially drove approximately straight.
- At the first major turn, the model did not apply sufficient steering.
- The vehicle continued straight, left the track and entered the water.

Likely issue:
The predicted steering values appear too weak for significant corners.

Changes considered:
- Investigate the magnitude of the predicted steering values.
- Test a steering gain before retraining.
- If necessary, add more targeted corner data or adjust model/training parameters.

Next test:
- Reduce target speed to 8.
- Keep steering gain at 1.5.
- Determine whether the second-turn failure is caused by insufficient
reaction time or inadequate steering prediction.
### Run 1 - Final Simulator Result

Training:
- Balanced samples: 1,649
- Training samples: 1,319
- Validation samples: 330
- Best epoch: 6
- Best validation loss: 0.1697

Simulator testing:

Test 1:
- Target speed: 12
- Steering gain: 1.0
- Result: Failed at first turn and entered the water.

Test 2:
- Target speed: 12
- Steering gain: 1.5
- Result: Passed the first turn but failed at the second turn.

Test 3:
- Target speed: 8
- Throttle: 0.15
- Steering gain: 1.5
- Result: Failed at the second turn.

Test 4:
- Target speed: 8
- Throttle: 0.15
- Steering gain: 2.0
- Result: Failed at the second turn.

Conclusion:
Increasing steering gain improved the first-turn response, but the vehicle
continued to fail at the second turn. Reducing speed did not resolve the
problem. This suggests that the CNN did not learn a sufficiently strong or
accurate representation of the second corner.

Run 2 will therefore focus on improving the training data rather than
further increasing the steering gain.

### Simulator Test - Track 2
Status: Not tested yet

Observations:
- To be completed after autonomous driving test.

### Changes Planned for Run 2
To be decided based on simulator performance.

