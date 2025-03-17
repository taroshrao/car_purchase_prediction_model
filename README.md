Car purchase amount prediction using ANN

Let's go through this step by step so that a beginner can understand it easily. We will break it down into small parts and explain what each section of the code does.

---

## **STEP 0: Import Libraries**
Before we can do anything, we need to import the necessary Python libraries.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
```
### **What do these libraries do?**
- `pandas` → Helps in handling and analyzing structured data (tables).
- `numpy` → Provides mathematical operations for working with arrays and matrices.
- `matplotlib.pyplot` → Used for plotting graphs and visualizing data.
- `seaborn` → Makes statistical visualizations easier and more attractive.
- `io` → Allows us to handle input and output operations (not used in the rest of the code).

---

## **STEP 1: Import Data**
We need to load the dataset, which contains customer information and car purchase amounts.

```python
import gdown
import pandas as pd

# Google Drive file ID extracted from your link
file_id = "1AjdB-C0Wrapujob1563RrcC-N7WWAzZ-"
output = "Car_Purchasing_Data.csv"

# Download the file
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Load the CSV file
car_df = pd.read_csv(output, encoding='ISO-8859-1')

# Display first few rows
car_df.head()
```

### **What happens here?**
1. We import `gdown`, which is used to download files from Google Drive.
2. We specify the `file_id` of the dataset and download it.
3. We load the dataset into a `pandas` DataFrame using `pd.read_csv()`.
4. We display the first few rows using `car_df.head()`.

---

### **Checking the Data**
```python
car_df.head(10)  # Shows the first 10 rows
car_df.tail(10)  # Shows the last 10 rows
```
- `car_df.head(10)` → Displays the first 10 rows of the dataset.
- `car_df.tail(10)` → Displays the last 10 rows.

---

## **STEP 2: Data Visualization**
Before training our model, we analyze the dataset using a scatterplot.

```python
sns.pairplot(car_df)
```
- This command creates a pairplot, which helps us visualize the relationships between different columns in the dataset.
- This step helps us understand which factors might affect the `Car Purchase Amount`.

---

## **STEP 3: Preparing the Data (Cleaning & Splitting)**
Now, we prepare the data by removing unnecessary columns and scaling the values.

### **1. Dropping Unnecessary Columns**
```python
X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis=1)
```
- We remove non-numerical data (`Customer Name`, `Customer e-mail`, `Country`) since these don't help in predicting car purchase amounts.
- The remaining columns (like age, salary, credit card debt, net worth) are stored in `X`, which will be used as input features.

```python
y = car_df['Car Purchase Amount']
```
- We store the target variable (`Car Purchase Amount`) in `y`, which will be predicted.

---

### **2. Checking Shapes of X and y**
```python
X.shape
y.shape
```
- `.shape` tells us how many rows and columns we have in `X` and `y`.

---

### **3. Scaling the Features**
Machine learning models work better when data is scaled to a small range (e.g., 0 to 1).

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```
- `MinMaxScaler()` scales all values between 0 and 1.
- `.fit_transform(X)` applies scaling to `X`.

We do the same for `y`:

```python
y = y.values.reshape(-1,1)  # Reshape y to make it compatible
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y)
```
- We reshape `y` because `MinMaxScaler` expects a 2D array.

---

## **STEP 4: Training the Model**
Now, we train a neural network to predict the car purchase amount.

### **1. Splitting the Data**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25)
```
- We split the data into **training (75%)** and **testing (25%)** sets.
- The model learns from `X_train` and `y_train`.
- We test its performance on `X_test` and `y_test`.

---

### **2. Creating a Neural Network**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential([
    Input(shape=(5,)),  # 5 input features
    Dense(40, activation='relu'),
    Dense(40, activation='relu'),
    Dense(1, activation='linear')
])
```
- We use **TensorFlow** to create a **neural network**.
- **Layers of the Model:**
  - Input Layer: Takes 5 features.
  - Hidden Layers: Two layers with **40 neurons each** using the `relu` activation function.
  - Output Layer: One neuron with **linear activation** (for regression).

```python
model.summary()
```
- Displays the model architecture.

---

### **3. Compiling and Training**
```python
model.compile(optimizer='adam', loss='mean_squared_error')
```
- `adam` optimizer helps in adjusting weights efficiently.
- `mean_squared_error` is used as the loss function (since this is a regression task).

```python
epochs_hist = model.fit(X_train, y_train, epochs=100, batch_size=75, verbose=1, validation_split=0.2)
```
- We train the model for **100 epochs** (iterations).
- A batch size of **75** means the model processes 75 samples at a time.
- `validation_split=0.2` → 20% of training data is used for validation.

---

## **STEP 5: Evaluating the Model**
Now, we check how well the model performed.

### **1. Plotting Loss Progress**
```python
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()
```
- This graph shows how the loss decreases over epochs.
- If the validation loss starts increasing while training loss keeps decreasing, it indicates **overfitting**.

---

### **2. Making Predictions**
We now use the trained model to predict a **new customer's car purchase amount**.

```python
X_test = np.array([[1, 50, 50000, 10000, 60000]])
y_predict = model.predict(X_test)

print('Expected Purchase Amount', y_predict)
```
- We input the values `[1, 50, 50000, 10000, 60000]`:
  - Gender: **1 (Male)**
  - Age: **50**
  - Annual Salary: **50,000**
  - Credit Card Debt: **10,000**
  - Net Worth: **60,000**
- The model predicts how much this person would spend on a car.

---

## **Conclusion**
1. We **imported the dataset** and **visualized it**.
2. We **cleaned and scaled** the data for better performance.
3. We **built and trained** a **neural network**.
4. We **evaluated** the model and **made predictions**.

This is how machine learning is used to predict car purchase amounts based on customer details. 🚀
