import random
import sys
import uuid
import warnings
from collections import OrderedDict
import NeuralNetwork as NN
import TNC_IBI
import LB_IBS
import CSVData
import socket
import pickle
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from torch.utils.data import DataLoader
import random
from ecdsa import NIST256p, ellipticcurve

# IBI Setup
curve = NIST256p
order = curve.order
G = curve.generator
client_TNC_IBI = TNC_IBI.TNC_ECC_Module(G, order)
# Client identifier
client_uuid = str(uuid.uuid4()) # Generate unique identifier, sumulation only
# Model training
local_model = NN.NeuralNetwork(17)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(local_model.parameters(), lr=0.01)
LOCAL_EPOCHS = 2
BATCH_SIZE = 32


warnings.filterwarnings("ignore", category=UserWarning, message="degrees of freedom is <= 0")

# Helper functions

def select_data(data, n: int, i: int):
    """
    Simulates data partitioning.

    Provided that clients are simulated altogether, to ensure that they all are assigned an equal fraction of different records 
    we use this procedure to select both features and labels from the given dataset. 

    ### Parameters:
    - data - Dataset to be used.
    - n - Total number of partitions.
    - i - The fraction index attatched to this client thread.

    ### Outputs

    A (features, labels) tuple from a selected fraction. 
    """
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    total_records = len(X)
    fraction_size = total_records // n
    r = total_records % n
    fractions = []
    start_idx = 0

    for j in range(n):
        end_idx = start_idx + fraction_size + (1 if j < r else 0)
        fraction_X = X[start_idx:end_idx]
        fraction_y = y[start_idx:end_idx]
        fractions.append((fraction_X, fraction_y))
        start_idx = end_idx

    if i < 1 or i > n:
        raise IndexError(f"Fraction index i={i} is out of range (1 to {n})")

    selected_fraction_X, selected_fraction_y = fractions[i - 1]  # Access the i-th fraction

    return selected_fraction_X, selected_fraction_y

def trainLocalModel(model, train_loader, client_id):
    """
    Federated local model training using partitioned data.
    """
    for epoch in range(LOCAL_EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"(CLIENT {client_id}) Epoch: {epoch + 1}, Loss: {train_loss: .3f}")

def evaluateLocalModel(model, test_loader):
    model.eval()
    test_loss = 0
    predictions = []
    labels = []
    with torch.no_grad(): 
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            # Move to CPU and convert to numpy
            predictions.extend(predicted.cpu().numpy())  
            labels.extend(y_batch.cpu().numpy())  
    test_loss /= len(test_loader) # Average test loss
    accuracy = accuracy_score(labels, predictions) 

    print(f"Local test Loss: {test_loss: .3f}")
    print(f"Local test Accuracy: {accuracy: .3f}")

    return test_loss, accuracy

# Attack section
# Inner Product Manipulation
def attack_IPM(weights, train_loader, byz_count, epsilon=15): # ensure that epsilon > K/M - 1 | 21/19 - 1 for a change of sign
    # Create a temporary local model
    local_model = NN.NeuralNetwork(17)
    local_model.load_state_dict(weights)
    honest_gradients = []

    # Collect honest gradients by training on the honest data
    for inputs, labels in train_loader:
        inputs, labels = inputs.to("cpu"), labels.to("cpu")
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.1) # 0.5
        optimizer.zero_grad() # comment if errors?
        outputs = local_model(inputs)
        loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        # Capture gradients for each parameter
        honest_gradient = {name: param.grad.clone() for name, param in local_model.named_parameters() if
                           param.grad is not None}
        honest_gradients.append(honest_gradient)

    # Average honest gradients to find the "honest direction"
    avg_honest_gradient = OrderedDict()
    for name in honest_gradients[0].keys():
        avg_honest_gradient[name] = torch.mean(torch.stack([g[name] for g in honest_gradients]), dim=0)

    # Generate poisoned gradients by maximizing the inner product deviation
    poisoned_weights = OrderedDict()
    for name, param in weights.items():
        if name in avg_honest_gradient:
            poisoned_weights[name] = weights[name] - epsilon * avg_honest_gradient[name] # Calculate poisoned update
        else:
            poisoned_weights[name] = weights[name]

    return poisoned_weights

# Gaussian Noise
def attack_noise(weights, mean, std):
    poisoned_weights = weights.copy()
    for p_name, p_tensor in weights.items():
        poison = random.gauss(mean, std)
        poisoned_weights[p_name] = p_tensor + poison

    return poisoned_weights

# A little is enough (ALIE)
def attack_ALIE(weights, z_max):

    poisoned_weights = weights.copy()
    for p_name, p_tensor in weights.items():
        if isinstance(p_tensor, torch.Tensor):
            mean = p_tensor.mean()
            std = p_tensor.std(unbiased=False)
            std_float = std.float()

            if torch.isclose(std_float, torch.tensor(0.), atol=1e-8): # avoid 0 values
                std = torch.tensor(1e-8, dtype=p_tensor.dtype)
            else:
                std = std_float

            poisoned_weights[p_name] = p_tensor - (z_max * std)
        else:
            poisoned_weights[p_name] = p_tensor

    return poisoned_weights

# Client connection
def main(local_id, atk_opt):
    """
    Clients main running loop, coordinates with the aggregator.
    
    ### Parameters:
    - local_id - Simulated id on a machine that runs multiple clients.
    - atk_opt - Attack option, chooses which attack to go with when injecting model poisoning. (0: gaussian noise, 1: ALIE, 2: IPM)
    """
    print(f"Client Started, ID: {local_id}")

    dataContainer = pd.read_csv('split_19.csv', sep=",")
    X, y = select_data(dataContainer, 9, int(local_id)) # separate the data equally among all simulated clients on the virtual machine
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    # Scale data
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    # Convert to data loaders
    train_dataset = CSVData.CSVDataset(X_train, y_train)
    test_dataset = CSVData.CSVDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"(CLIENT {local_id}) Data loaded and prepared successfully.")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('10.0.2.7', 12345))

        # IBI Client Authentication
        # Inform of identity
        auth_dict = {"ibi_identity": client_uuid}
        auth_data = pickle.dumps(auth_dict)
        s.sendall(auth_data)
        print(f"Community identity ({client_uuid}) to the aggregator.")

        response_uks = bytearray()
        while True:
            raw_data = s.recv(1024)
            if not raw_data:
                break
            if b'end_chunk' in raw_data:
                response_uks.extend(raw_data)
                break
            response_uks.extend(raw_data)

        buffer = BytesIO(response_uks)
        uks = pickle.load(buffer)
        print(f"Loading keys from the aggregator for client ID: {client_uuid}")

        # IBI
        k_ibi = uks[0]
        uk_ibi = k_ibi[0]
        mpk_ibi = k_ibi[1]
        # IBS 
        k_ibs = uks[1]
        uk_ibs = k_ibs[0]
        mk_ibs = k_ibs[1]
        q_ibs = k_ibs[2]
        LB_IBS_Client = LB_IBS.LB_IBI_Module(q_ibs)

        client_TNC_IBI.pk = mpk_ibi  # store master public key (mpk)
        s.sendall(b'ACK')

        # Interactively prove clients identity, pg 9: https://doi.org/10.3390/sym13081330
        # Verify one's own identity in AuthMode to get U' and V'
        print(f"(CLIENT {client_id}) Authentication Request Acknowledged.")
        UV_prime = client_TNC_IBI.Verify(client_uuid, uk_ibi, AuthMode=True)
        t = random.randint(1, order - 1)
        T = G * t
        UVT = (UV_prime[0], UV_prime[1], T)
        UVT_encoded = pickle.dumps(UVT)

        print(f"Sending commitment to the aggregator for client ID: {client_uuid}")
        s.sendall(UVT_encoded)  # sends (U', V', T) for authentication

        # Solve challenge as proof, transmit it to the verifier (Aggregator)
        challenge_data = s.recv(1024)
        try:
            c = int(pickle.loads(challenge_data))
        except Exception as e:
            print(f"(CLIENT {local_id}) Failed to receive a valid challenge. Error: {e}")
            s.close()
            return
        print(f"(CLIENT {client_uuid}) Challenge received. Computing solution.")
        e = (t + c * uk_ibi[0]) % order
        response_dict = {"response_e": e}
        response_data = pickle.dumps(response_dict)
        print(f"(CLIENT {client_uuid}) Solution computed. Transferring to aggregator.")
        s.sendall(response_data)

        s.recv(1024) # ACK Sequence to start clients FL process
        s.sendall(b'ACK') 

        raw_weights = bytearray()
        while True:
            try:
                raw_data = s.recv(1024)
                if not raw_data:
                    break
                if b'susb.' in raw_data:
                    raw_weights.extend(raw_data)
                    break
                raw_weights.extend(raw_data)
            except socket.timeout:
                break

        print(f"(CLIENT {local_id}) Model received. Loading parameters...")
        buffer = BytesIO(raw_weights)
        weights = pickle.load(buffer)
        local_model.load_state_dict(weights)

        fed_round = 1
        while True:
            try:
                print(f"(CLIENT {local_id}) Proceeding with model training. Round: {fed_round}")
                trainLocalModel(local_model, train_loader, local_id)
                torch.save(local_model.state_dict(), f'local_model_weights_{local_id}.pth')
                feedback_weights = torch.load(f'local_model_weights_{local_id}.pth', weights_only=True)

                # Attack Injection
                # Noise attack (using gaussian distribution)
                if atk_opt == 0:
                    print(f"(CLIENT {client_id}) Injecting Poison")
                    poisoned_weights = attack_noise(feedback_weights, 2, 2)
                # A Little Is Enough
                if atk_opt == 1:
                    print(f"(CLIENT {client_id}) Injecting Poison")
                    poisoned_weights = attack_ALIE(feedback_weights, 0.9) # (Use standard nominal distribution table)
                # IPM attack
                if atk_opt == 2:
                    poisoned_weights = attack_IPM(feedback_weights, train_loader, 9)
                # Sign and send
                signature = LB_IBS_Client.Sign(mk_ibs, uk_ibs, str(poisoned_weights), client_uuid)
                signed_weights = (poisoned_weights, signature, b'end_chunk')
                raw_feedback = pickle.dumps(signed_weights)


                # Unpoisoned
                # feedback_weights["end"] = 0
                # raw_feedback = pickle.dumps(feedback_weights)

                s.sendall(raw_feedback)

                # data = s.recv(1024).decode()
                # print(data)

                # Client receiving global weights (global model) from the server
                raw_weights = bytearray()
                while True:
                    try:
                        raw_data = s.recv(1024)
                        if not raw_data:
                            break
                        if b'end\x94K\x00u.' in raw_data or b'susb.' in raw_data: 
                            raw_weights.extend(raw_data)
                            break
                        raw_weights.extend(raw_data)
                    except socket.timeout:
                        break

                print(f"(CLIENT {local_id}) New global model received. Loading parameters.")
                buffer = BytesIO(raw_weights)
                weights = pickle.load(buffer)
                del weights["end"]
                local_model.load_state_dict(weights)

                print(f"(CLIENT {local_id}) Proceeding with model evaluation.")
                loss, accuracy = evaluateLocalModel(local_model, test_loader)
                accuracy = round(accuracy, 5)
                s.sendall(str(loss).encode())

                final_loss = bytearray()
                while True:
                    try:
                        raw_data = s.recv(1024)
                        final_loss.extend(raw_data)
                        break
                    except socket.timeout:
                        break
                final_loss_string = final_loss.decode()
                print(final_loss_string)

                s.sendall(str(accuracy).encode())
                fed_round += 1

            except BrokenPipeError as e:
                s.close()
                main(local_id, 1)
                break
            except EOFError as e:
                print(f"(CLIENT {local_id}) Error: Connection refused... Attempting to reconnect")
                s.close()
                main(local_id, 1) # Simulate reconnections, try the main loop one more time to see if the aggregator is RMC resistant
                break
    print("aaa")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        client_id = sys.argv[1]
        main(client_id, 0)
    else:
        print("Client ID not passed.")