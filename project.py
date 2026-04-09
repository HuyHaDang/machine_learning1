import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. LOAD AND PREP DATA
df = pd.read_csv("online_gaming_behavior_dataset.csv")

# Create Label: 1 if "High" engagement, else 0
df["addicted"] = df["EngagementLevel"].apply(lambda x: 1 if x == "High" else 0)

# Select Features
features = [
    "Age",
    "PlayTimeHours",
    "SessionsPerWeek",
    "AvgSessionDurationMinutes",
    "InGamePurchases"
]

X_raw = df[features]
y_raw = df["addicted"]

# Train/Test Split (80% Train, 20% Test)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42)

print("Train:", X_train_raw.shape)
print("Test:", X_test_raw.shape)

# 2. NORMALIZE & ADD BIAS
# Calculate mean and std strictly on the TRAINING set to prevent data leakage
train_mean = X_train_raw.mean()
train_std = X_train_raw.std()

# Normalize Train and Test using the TRAIN mean/std
X_train = X_train_raw.apply(lambda rec: (rec - train_mean) / train_std, axis=1)
X_test = X_test_raw.apply(lambda rec: (rec - train_mean) / train_std, axis=1)

# Add x0 equal to 1 using np.c_
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Convert y targets to NumPy arrays
Y_train = y_train.values
Y_test = y_test.values

# 3. CORE LOGISTIC LOGIC
def initialize(dim):
    theta = np.random.rand(dim)
    return theta #size(3,)

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def predict_Y(theta, X):
    return sigmoid(np.dot(X, theta)) #size(20000,)

def cost_function(theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(predict_Y(theta, x)) + (1 - y) * np.log(
            1 - predict_Y(theta, x) ))
    return total_cost

def update_theta(x, y, y_hat, theta_o, learning_rate):
    dw = (np.dot((y_hat - y), x) * 2) / len(y) #size of dw (6,)
    theta_1 = theta_o - learning_rate * dw
    return theta_1

# 4. TRAINING LOOP
def run_gradient_descent(X, Y, alpha, num_iterations):
    theta = initialize(X.shape[1])
    iter_num = 0
    gd_iterations_df = pd.DataFrame(columns=['iteration', 'cost'])
    result_idx = 0
    
    for each_iter in range(num_iterations):
        Y_hat = predict_Y(theta, X)
        this_cost = cost_function(theta, X, Y)
        prev_theta = theta
        theta = update_theta(X, Y, Y_hat, prev_theta, alpha)
        
        if (iter_num % 10 == 0):
            gd_iterations_df.loc[result_idx] = [iter_num, this_cost]
            result_idx = result_idx + 1
        iter_num += 1
        
    print("Final Estimate of theta : ", theta)
    return gd_iterations_df, theta

# Run the training
gd_iterations_df, theta = run_gradient_descent(X_train, Y_train, alpha=0.01, num_iterations=4000)

# 5. VISUALIZATION
plt.plot(gd_iterations_df['iteration'], gd_iterations_df['cost'])
plt.xlabel("Number of iterations")
plt.ylabel("Cost or MSE")
plt.show()

# 6. EVALUATION
# Test Set
Y_hat_test = predict_Y(theta, X_test)
print("\n=== FINAL TEST RESULT ===")
print("Test Accuracy:", accuracy_score(Y_hat_test.round(), Y_test))
# print("Confusion Matrix:")
# print(confusion_matrix(Y_test, Y_hat_test.round()))

# =========================
# 7. PREDICT NEW USER
# =========================
# print("\n=== NEW USER PREDICTION ===")
# # Age=25, PlayTime=6h, Sessions=5, Duration=120min, Money=30
# new_user_df = pd.DataFrame([[25, 6, 5, 120, 30]], columns=features)

# # normalize using SAME mean/std from Training
# new_user_norm = new_user_df.apply(lambda rec: (rec - train_mean) / train_std, axis=1)

# # add bias
# new_user_final = np.c_[np.ones((1, 1)), new_user_norm]

# prob = predict_Y(theta, new_user_final)[0]

# print("Probability of addiction:", prob)
# if prob >= 0.5:
#     print("Prediction: ADDICTED")
# else:
#     print("Prediction: NOT ADDICTED")



from sklearn.linear_model import LogisticRegression

# 8. COMPARE WITH SCIKIT-LEARN
print("\n=== SCIKIT-LEARN BENCHMARK ===")

sk_model = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=500)

sk_model.fit(X_train, Y_train)
# Predict on the test set
sk_predictions = sk_model.predict(X_test)
# Evaluate
print(f"Scikit-Learn Test Accuracy: {accuracy_score(sk_predictions, Y_test):.4f}")
print("Custom Algorithm Accuracy: ", accuracy_score(Y_hat_test.round(), Y_test))
