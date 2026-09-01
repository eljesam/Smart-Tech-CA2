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

## Run 2 - Targeted Corner Data

Changes from Run 1:
- Added targeted training data from the second corner on Track 1
- Added recovery examples around the problematic corner
- Changed balancing strategy to preserve turning samples while reducing
  excessive near-straight samples
- CNN architecture and major training hyperparameters remained unchanged

Training result:
- Training stopped after approximately 5 epochs
- Best validation loss: approximately 0.123
- Best validation performance occurred around Epoch 2

Comparison:
Run 1 best validation loss: 0.1697
Run 2 best validation loss: ~0.123

Observation:
The additional targeted corner data produced a substantial reduction in
validation error without increasing model complexity.

Settings:
- Target speed: 8
- Throttle: 0.15
- Steering gain: 1.0
- Steering bias: 0.0

Result:
- The vehicle successfully passed the first turn.
- The vehicle also successfully passed the second turn that caused Run 1 to fail.
- After the second turn, the vehicle showed a persistent tendency to remain too far left.
- On the straight section, the vehicle gradually moved further left and eventually left the track.

Observation:
Run 2 substantially improved cornering performance compared with Run 1,
but introduced a left-steering bias on straighter sections.

### Simulator Test 2
Settings:
- Target speed: 8
- Throttle: 0.15
- Steering gain: 1.0
- Steering bias: +0.03

Result:
- The vehicle failed at the second turn again.

Observation:
Adding a positive steering bias reduced the model's ability to make the
required left turn.

### Simulator Test 3
Settings:
- Target speed: 8
- Throttle: 0.15
- Steering gain: 1.0
- Steering bias: +0.05

Result:
- The vehicle again failed at the second turn.

Observation:
Increasing the steering bias further did not solve the straight-road drift
and negatively affected cornering performance.

### Run 2 Conclusion
Run 2 was a clear improvement over Run 1 because the model learned to
successfully negotiate both of the first two major corners. However, the
model developed a persistent left bias on straight sections.

Attempts to correct this using a fixed steering bias caused the model to
under-steer at the second turn. This indicated that a fixed correction in
drive.py was not an appropriate solution.

Run 3 therefore focuses on retaining more near-straight training examples
while preserving the improved corner data from Run 2.

### Run 3 - Simulator Test 1

Settings:
- Target speed: 8
- Throttle: 0.15
- Steering gain: 1.0
- Steering bias: 0.0

Result:
- Vehicle eventually left the road and became stuck on the grass.

Observations:
- First turn: TBD
- Second turn: TBD
- Straight section: TBD
- Failure location: TBD

## Run 3 - Increased Straight-Driving Samples

### Changes from Run 2
- Retained more near-straight driving samples during dataset balancing.
- Increased maximum retained near-straight samples from 1,000 to 1,800.
- Kept the CNN architecture, augmentation and other training parameters unchanged.

### Training Results
- Best validation loss: approximately 0.098
- This was lower than both Run 1 and Run 2.

### Simulator Test
Settings:
- Target speed: 8
- Throttle: 0.15
- Steering gain: 1.0
- Steering bias: 0.0

Result:
- The vehicle successfully passed the first turn.
- On approaching the second turn, the vehicle continued approximately straight.
- The vehicle failed to apply sufficient steering and left the track.

### Conclusion
Although Run 3 achieved the lowest validation loss, its simulator performance
was worse than Run 2 at the second corner.

Increasing the number of straight-driving samples reduced the previous
left-steering bias but also reduced the model's responsiveness to the
important corner.

This demonstrates that validation loss alone is not sufficient to assess
the driving model. Simulator testing is also required.

Run 4 will use an intermediate number of near-straight samples in an attempt
to retain both the improved cornering behaviour of Run 2 and better
straight-road stability.

