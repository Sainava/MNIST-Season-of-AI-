
## Handwritten Digit Recognition — MNIST with TensorFlow

This project demonstrates a simple neural network using **TensorFlow** to classify handwritten digits from the popular **MNIST dataset**.

---

###  **Project Overview**

* **Dataset:** MNIST — 70,000 grayscale images of handwritten digits (0–9).
* **Model:**

  * Input: 28×28 pixel images, flattened to a vector.
  * Architecture:

    * Flatten layer
    * Dense layer with 128 ReLU units
    * Second Dense layer with 128 ReLU units
    * Output Dense layer with 10 softmax units (one for each digit)
  * Optimizer: Adam
  * Loss Function: Sparse Categorical Crossentropy
  * Training Epochs: 3

---

###  **Features**

* Loads and normalizes MNIST data.
* Builds and trains a simple fully connected neural network.
* Evaluates the model on test data.
* Saves the trained model for reuse.

---

###  **Requirements**

Make sure you have:

* Python 3.x
* TensorFlow
* OpenCV (`cv2`)
* NumPy
* Matplotlib

Install dependencies with:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

---

###  **How to Run**

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. **Run the notebook**

   Open the notebook in Jupyter or any IDE that supports `.ipynb` files.

   Or run as a Python script:

   ```bash
   python mnist_model.py
   ```

3. **Train the model**

   The notebook automatically trains for 3 epochs.

4. **Save & Load**

   After training, the model is saved as `HandWrittneDigits.keras`.
   Load it later for evaluation or inference.

---

###  **Sample Results**

* Training Accuracy: \~97.8%
* Test Accuracy: \~97%
* Saved model: `HandWrittneDigits.keras`

---

###  **File Structure**

```
.
├── mnist_model.ipynb   # Main notebook
├── HandWrittneDigits.keras  # Saved model after training
```

---

###  **Possible Extensions**

* Add convolutional layers for improved accuracy.
* Visualize predictions on new handwritten samples.
* Build a simple web interface to test your own handwriting.

---

###  **License**

MIT License.
Feel free to fork and experiment!

