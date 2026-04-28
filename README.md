# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1650" height="848" alt="Screenshot 2026-04-21 113451" src="https://github.com/user-attachments/assets/d6caca5b-e789-43e3-88d1-142f2f12ca05" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: LAAVANYA.R

### Register Number: 212224230135

```class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
       elf.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
   for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        model.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")



train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)



with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")



loss_df = pd.DataFrame(ai_brain.history)

loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


X_new = torch.tensor([[9]], dtype=torch.float32)

X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

prediction = ai_brain(X_new_tensor).item()

print(f"Prediction: {prediction}")

```

### Dataset Information
<img width="207" height="454" alt="Screenshot 2026-04-28 102847" src="https://github.com/user-attachments/assets/245984eb-fd8b-4c72-8ec5-aed00d7361a1" />


### OUTPUT
<img width="511" height="236" alt="Screenshot 2026-04-28 103139" src="https://github.com/user-attachments/assets/5d7ae4d8-e548-47ee-87a7-2341605d7a8f" />



### Training Loss Vs Iteration Plot
<img width="883" height="586" alt="Screenshot 2026-04-28 103223" src="https://github.com/user-attachments/assets/850fc6b9-1c66-4cbe-ae1e-df9fd8c96b86" />


### New Sample Data Prediction!

<img width="430" height="56" alt="Screenshot 2026-04-28 103810" src="https://github.com/user-attachments/assets/a28f7502-8e0d-4cbe-a271-c8bbbccc50ff" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
