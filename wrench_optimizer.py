import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#We first Load and preprocess CSV ===
df = pd.read_csv('/home/yassine/Documents/code/co2_mm_mlo.csv', comment='#')
df = df[df['average'] > 0]
df['months_since_start'] = (df['year'] - df['year'].min()) * 12 + (df['month'] - 1)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

#We begin by extracting features and targets
X = df['months_since_start'].values.reshape(-1, 1)
y = df['average'].values

#We normalize the data, by the standardization formula: (x - mean) / (standard deviation)
X_mean = X.mean()
X_std = X.std()
X_norm = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y_norm = (y - y_mean) / y_std

# We create a function for the gradient computation ===
def compute_gradients(X, y, w, b):
    n = len(y)
    
    prediction = np.zeros(n)
    for i in range(n):
        prediction[i] = np.dot(X[i], w) + b
    
    dw = np.zeros_like(w)
    for j in range(len(w)):
        grad_sum = 0.0
        for i in range(n):
            grad_sum += X[i,j] * (prediction[i] - y[i])
        dw[j] = grad_sum / n
    
    db_sum = 0.0
    for i in range(n):
        db_sum += prediction[i] - y[i]
    db = db_sum / n
    
    return dw, db

#The Standard Gradient Descent:
def gradient_descent(X, y, w_init, b_init, lr, n_iter):
    w = w_init.copy()
    b = b_init
    losses = []
    for _ in range(n_iter):
        dw, db = compute_gradients(X, y, w, b)
        w -= lr * dw
        b -= lr * db
        
        # Compute loss
        loss_sum = 0.0
        for i in range(len(y)):
            pred_i = np.dot(X[i], w) + b
            loss_sum += np.power(pred_i - y[i], 2)
        loss = loss_sum / len(y)
        losses.append(loss)
    return w, b, losses

#Our Wrench-Coupled Optimizer Function:
def wrench_coupled_optimizer(X, y, w_init, b_init, m, I, n_iter):
    # Here: m = mass, I = moment of inertia
    w = w_init.copy()
    b = b_init  # this is the initial bias parameter
    losses = []
    
    for iteration in range(n_iter):
        dw, db = compute_gradients(X, y, w, b)
        
        # w(t+1) = w(t) - (1/m) * ∂J/∂w - (1/I) * ∂J/∂b
        new_w = np.zeros_like(w)
        for j in range(len(w)):
            new_w[j] = w[j] - (1.0 / m) * dw[j] - (1.0 / I) * db
        
        # b(t+1) = b(t) - (1/m) * ∂J/∂b + (1/I) * sum(∂J/∂w)
        sum_dw = 0.0
        for j in range(len(dw)):
            sum_dw += dw[j]
        new_b = b - (1.0 / m) * db + (1.0 / I) * sum_dw
        
        w = new_w
        b = new_b

        #Now, we compute loss
        loss_sum = 0.0
        for i in range(len(y)):
            pred_i = np.dot(X[i], w) + b
            loss_sum += np.power(pred_i - y[i], 2)
        loss = loss_sum / len(y)
        losses.append(loss) #this is to be later used to show the loss function
        
    return w, b, losses

# Now comes the training:
np.random.seed(0)
w_init = np.random.randn(1)
b_init = np.random.randn()  

n_iter = 1000
lr = 0.01
m = 5.0   
I = 20.0  


w_gd, b_gd, losses_gd = gradient_descent(X_norm, y_norm, w_init, b_init, lr, n_iter)
w_wc, b_wc, losses_wc = wrench_coupled_optimizer(X_norm, y_norm, w_init, b_init, m, I, n_iter)



plt.figure(figsize=(12,8))
plt.plot(losses_gd, label='Gradient Descent', color='blue', linewidth=2)
plt.plot(losses_wc, label='Wrench-Coupled Optimizer', color='red', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss (Log Scale)')
plt.title('Loss Convergence Comparison')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()


print("Initial loss - GD: %.6f, WC: %.6f" % (losses_gd[0], losses_wc[0]))
print("Final loss - GD: %.6f, WC: %.6f" % (losses_gd[-1], losses_wc[-1]))
print("Loss difference at end: %.8f" % abs(losses_gd[-1] - losses_wc[-1]))

# Next would be denormalizing the params: the formula: original = normalized * std + mean ===
def denormalize_params(w, b, X_mean, X_std, y_mean, y_std):
    w_orig = w * (y_std / X_std)
    b_orig = b * y_std + y_mean - w_orig * X_mean
    return w_orig, b_orig

w_gd_orig, b_gd_orig = denormalize_params(w_gd, b_gd, X_mean, X_std, y_mean, y_std)
w_wc_orig, b_wc_orig = denormalize_params(w_wc, b_wc, X_mean, X_std, y_mean, y_std)

print("Gradient Descent final params: w=%.4f, b=%.4f" % (w_gd_orig[0], b_gd_orig))
print("Wrench-Coupled final params: w=%.4f, b=%.4f" % (w_wc_orig[0], b_wc_orig))

# We finally visualize the predictions:
X_months = df['months_since_start'].values
y_pred_gd = w_gd_orig[0] * X_months + b_gd_orig
y_pred_wc = w_wc_orig[0] * X_months + b_wc_orig

plt.figure(figsize=(14, 8))
plt.plot(df['date'], df['average'], 'o', label='Actual CO2 levels', color='black')
plt.plot(df['date'], y_pred_gd, label='Gradient Descent', color='blue', linewidth=2)
plt.plot(df['date'], y_pred_wc, label='Wrench-Coupled', color='red', linewidth=2)
plt.xlabel('Date')
plt.ylabel('CO2 Concentration (ppm)')
plt.title('CO2 Concentration Predictions')
plt.legend()
plt.grid(True)
plt.show()