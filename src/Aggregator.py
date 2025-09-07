import csv
import math
import socket
import threading
import pickle
import NeuralNetwork as NNlib
import TNC_IBI
import LB_IBS
from Crypto.Util.number import getPrime
import torch
import torch.nn as nn
import numpy as np
from io import BytesIO
# IBI Setup 
import random
from ecdsa import NIST256p, ellipticcurve
# Choose the NIST256p curve (secp256r1)
curve = NIST256p
order = curve.order
G = curve.generator
# IBI PREP
# Note: in IBI, setup phase done through MKGen() would be executed by a Trust Authority (TA). We combine these together for the simplicity purposes of experiment.
Aggregator_TNC_IBI = TNC_IBI.TNC_ECC_Module(G, order)
k = Aggregator_TNC_IBI.MKGen()
# IBS PREP
q = getPrime(34) # 90 bit security level for best knkown attack
Aggregator_LB_IBS = LB_IBS.LB_IBI_Module(q)
k_ibs = Aggregator_LB_IBS.Setup()
disconnected_identities = set() # A global set to keep track of identities that have been disconnected.

# Threading and Federated Learning
print_lock = threading.Lock()
round_lock = threading.Lock()
round_condition = threading.Condition(round_lock)
global_round = 1
round_finish_count = 0
active_clients_count = 0
FL_CHECK = 10

# Initialize weights
init_nn = NNlib.NeuralNetwork(17)
torch.save(init_nn.state_dict(), 'model_weights.pth')
# Connection and Simulation Parameters
HOST = "0.0.0.0"
PORT = 12345
EXPECTED_CLIENTS = 5 
BYZANTINE_CLIENTS = 14
FL_ROUNDS = 50
# Aggregator Paramaters - may be replaced with config file in the future
clients_connected = 0
awaiting_clients = 0
client_responses = 0
end_response_check = 0
all_weights = []
loss_list = []
accuracy_list = []
raw_feedback_check_list = []
client_weights_dict = {} # Holds a map of client addresses -> weights
client_record_dict = {} # Holds a map of clients to their respective records to track their reputation.
# Checks to eliminate repeated actions over multiple threads (Simulation purposes only)
rpcheck_once = True
rmcheck_once = True
isSaved = False

def aggregate():
	"""
	Default aggregation function. Based on FedSGD, averages all gradients.
	"""
	while True:
		if len(all_weights) < clients_connected:
			continue # Wait until all clients are connected
		else:
			aggregated_weights = {}
			for tensor in all_weights[0]:
				tensor_list = [weights[tensor].float() for weights in all_weights]
				aggregated_weights[tensor] = torch.stack(tensor_list).mean(dim=0)
			return aggregated_weights

def aggregate_krum(weights: list, f: int):
	"""
	### Krum Aggregation. 
	Solely for the experimental purposes. 

	### Parameters:
	- weights - List of all weights recieved from clients.
	- f - Number of assumed Byzantine clients the aggregator is ensured to tolerate, as per Krum definition.
	
	### Outputs

	A tuple consisting of:
		- aggregated_update - optimal vector deemed as most reliable by Krum aggregation rule.
		- excluded_clients - list of indexes for clients that have not been included in the aggregation (as such may be interpreted as not reliable enough for reputation purposes)
	"""
	while True:
		if len(all_weights) < active_clients_count - awaiting_clients:
			continue
		else:
			n = len(weights)
			flat_weights = []
			for grad in weights:
				if "end" in grad:
					del grad["end"] # Remove redundancies

				grad_values = list(grad.values())
				flat_tensors = []
				for param in grad_values:
					try:
						flat_tensors.append(param.flatten())
					except AttributeError:
						print(f"Warning. Detected int in parameters: {param}")
						continue
				flat_weights.append(torch.cat(flat_tensors))

			scores = []
			for i in range(n):
				# compute scores for each vector based on the euclidean distances
				distances = [
					torch.norm(flat_weights[i] - flat_weights[j]).item() ** 2
					for j in range(n) if j != i
				]
				closest_distances = sorted(distances)[:n - f - 2]
				scores.append(sum(closest_distances))

			krum_index = scores.index(min(scores))
			aggregated_update = weights[krum_index] # Establish the most reliable gradient/update

			# Get list of excluded clients
			excluded_clients = [i for i in range(n) if i != krum_index]

			return aggregated_update, excluded_clients

def aggregate_trmean(weights, trim_ratio=0.4):
	"""
	### Trimmed Mean Aggregation.
	Solely for the experimental purposes. 

	### Parameters:
	- weights - List of all weights recieved from clients.
	- trim_ratio - ratio of updates to be trimmed from aggregation on each end of the distribution.

	### Outputs

	A tuple consisting of:
	- aggregated_update - optimal vector deemed as most reliable by Krum aggregation rule.
	- excluded_clients - list of indexes for clients that have not been included in the aggregation (as such may be interpreted as not reliable enough for reputation purposes)
	"""
	while True:
		if len(all_weights) < active_clients_count - awaiting_clients:
			continue
		else:
			num_clients = len(weights)
			trim_count = int(trim_ratio * num_clients)

			aggregated_update = {}
			# Update indexes both tails from the update distribution
			excluded_clients = list(range(trim_count)) + list(range(num_clients - trim_count, num_clients)) 

			for param in weights[0]:
				# Stack all updates and sort them
				param_values = torch.stack([update[param] for update in weights], dim=0)
				sorted_values, indices = torch.sort(param_values, dim=0)

				# Trim out top and bottom clients
				trimmed_values = sorted_values[trim_count:num_clients - trim_count]
				aggregated_update[param] = trimmed_values.mean(dim=0) #

			return aggregated_update, excluded_clients

# Helper Functions
loss_results = []
accuracy_results = []

def compute_loss():
	while True:
		if len(loss_list) < active_clients_count - awaiting_clients:
			continue
		else:
			mean = np.mean(loss_list)
			return mean

def compute_accuracy():
	while True:
		if len(accuracy_list) < active_clients_count - awaiting_clients:
			continue
		else:
			mean = np.mean(accuracy_list)
			return mean

def updateLossTable(loss):
	loss_results.append(loss)

def updateAccuracyTable(accuracy):
	accuracy_results.append(accuracy)

def save_results():
	global accuracy_results, loss_results
	numeric_loss = [float(item) for item in loss_results]
	numeric_accuracy = [float(item) for item in accuracy_results]

	with open("RESULTS.csv", "w", newline="") as file:
		writer = csv.writer(file)
		writer.writerow(["Loss", "Accuracy"])

		for result1, result2 in zip(numeric_loss, numeric_accuracy):
			writer.writerow([result1, result2])

	print("Results Saved.")

def handle_client(connection, address):
	"""
	Main process behind the aggregator using python sockets. Handles authentication for each client and their collective processing during FL.

	Each handle_client function is processed on a thread for each individual client.   
	
	### Parameters:
	- connection - specifies the connection that the aggregator listens on, entry point for all potential clients.
	- address - specifies the address from which the connection is coming from.

	### Important
	The implemented procedure has been, and should be, used as proof of concept only. 
	"""

	global BYZANTINE_CLIENTS, FL_CHECK, clients_connected, global_round, round_finish_count, active_clients_count, awaiting_clients
	global client_responses, end_response_check, all_weights, rpcheck_once, rmcheck_once

	# IMPORTANT: the following procedure serves as proof of concept only, it is not adapted for real-world deployments.
	# It is recommended to handle secure implementations using more robust and efficient languages such as C. 
	authenticated_identity = None 
	try:
		"""
		Phase 1, Authentication: Handle recieved connections and produce keys based on the identities.
		"""
		auth_data = connection.recv(1024) # Recieve clients request to connect
		try:
			auth_info = pickle.loads(auth_data)
			received_identity = auth_info["ibi_identity"]
			# IBI
			uk_ibi = Aggregator_TNC_IBI.UKGen(received_identity)
			mpk_ibi = Aggregator_TNC_IBI.pk
			# IBS
			uk_ibs = Aggregator_LB_IBS.Extract(k_ibs[0], k_ibs[1], received_identity)
			uk_decoded = ((uk_ibi, mpk_ibi), (uk_ibs, k_ibs[0], q), b'end_chunk')

			uk_encoded = pickle.dumps(uk_decoded)
			conn.sendall(uk_encoded)
			ACK = connection.recv(1024)
			pickle.dumps(ACK)
			print(f"Authentication ACK from {address}")
		except Exception as e:
			print(e)
			print("Authentication failed: invalid data format")
			connection.close()
			return

		if received_identity in disconnected_identities:
			# Verify if identity hasn't been disconnected before.
			print("Authentication failed: identity has been permanently excluded.")
			connection.close()
			return

		# Authenticate Client
		UVT_response = connection.recv(1024)
		Client_UVT = pickle.loads(UVT_response)
		print(f"Commitment received from: {address}")
		# Challenge Generation
		c = random.randint(1, order - 1)
		challenge_msg = pickle.dumps(c)
		connection.sendall(challenge_msg)
		print(f"Challenge sent to: {address}")

		# Challenge verification
		response_data = connection.recv(1024)
		try:
			response_info = pickle.loads(response_data)
			e = response_info["response_e"]
		except Exception as e:
			print("Authentication failed: invalid response format")
			connection.close()
			return

		print(f"Challenge solution received from: {address}.")
		x_prime = Aggregator_TNC_IBI.Selected_H(received_identity, Client_UVT[0], Client_UVT[1])
		# Check if g^e == T(U'*y_1^x')^c
		left = G * e
		right = Client_UVT[2] + c*(Client_UVT[0] + x_prime*Aggregator_TNC_IBI.pk[0])
		if not (left.x() == right.x() and left.y() == right.y()):
			print("Authentication failed: proof verification failed")
			connection.close()
			return

		authenticated_identity = x_prime
		print(f"(IBI) Client {received_identity} authenticated successfully from {address}")

		with connection:
			"""
			Phase 2, Federated Learning: Proceed with handling client thread for FL.
			"""
			connection.sendall(b'ACK')
			_ = connection.recv(1024).decode() # Wait for ACK from client
			clients_connected += 1
			accepted = False
			awaiting_clients += 1
			with round_lock:
				# Handle dynamically connecting clients.
				if global_round == 1:
					local_round = 1
					active_clients_count += 1
					awaiting_clients -= 1
					accepted = True
					print(f"(ADDR {address}) Joining at initial round {local_round}")
				else:
					local_round = global_round + 1
					print(f"(ADDR {address}) Training in progress. Will join at round {local_round}")

			with round_condition:
				# Wait for the next round if joined mid-procedure.
				while global_round < local_round:
					round_condition.wait()

			weights = torch.load('model_weights.pth', weights_only=True)
			raw_weights = pickle.dumps(weights)
			connection.sendall(raw_weights)
			print(f"(ADDR {address}) Sent global model for round {global_round}")


			while local_round <= FL_ROUNDS:
				if not accepted:
					awaiting_clients -= 1
					accepted = True

				raw_feedback = bytearray()
				while True:
					try:
						chunk = connection.recv(1024)
						if not chunk:
							break
						if b'end_chunk' in chunk:
							raw_feedback.extend(chunk)
							break
						rpcheck_once = False # reset to false before breaking out to avoid de-synchronization issues
						rmcheck_once = False
						end_response_check = 0
						raw_feedback.extend(chunk)
					except socket.timeout:
						break

				client_responses += 1
				buffer = BytesIO(raw_feedback)
				signed_feedback_weights = pickle.load(buffer)
				feedback_weights = signed_feedback_weights[0]
				signature = signed_feedback_weights[1]

				# Reject and disconnect the client if message has been tampered with
				if not Aggregator_LB_IBS.Verify(k_ibs[0], signature, str(feedback_weights), received_identity):
					print(f"(Client {address}) Error: Signature doesn't match the message.")
					connection.close()
					break

				client_weights_dict[authenticated_identity] = feedback_weights

				while len(client_weights_dict) < (active_clients_count - awaiting_clients):
					continue
				# Aggregation
				all_weights = list(client_weights_dict.values())
				global_weights, excluded_clients = aggregate_krum(all_weights, BYZANTINE_CLIENTS)

				# Keep track of the clients exclusion from the aggregation
				if not rpcheck_once:
					rpcheck_once = True
					identities = list(client_weights_dict.keys())
					for idx, record_id in enumerate(identities):
						if record_id not in client_record_dict:
							client_record_dict[record_id] = []
						record_value = 1 if idx in excluded_clients else 0
						client_record_dict[record_id].append(record_value)
						if len(client_record_dict[record_id]) > FL_CHECK:
							client_record_dict[record_id].pop(0)

				# Forcible disconnection based on reputation
				removed_clients = 0
				"""
				Check the average value of arrays that hold exclusion instances for each address. 
				If the average == 1, then the client has failed to be deemed as trustworthy for aggregation even once
				in the specified number of rounds, hence disconnect.
				"""
				record = client_record_dict[authenticated_identity]
				avg = sum(record)/len(record)
				if avg == 1 and len(record) == FL_CHECK:
					print(f"Warning: Untrustworthy client detected. Proceeding removal from aggregation: {address}")
					try:
						del client_weights_dict[authenticated_identity]
						del client_record_dict[authenticated_identity]
						removed_clients += 1
						active_clients_count -= 1
						clients_connected -= 1
						disconnected_identities.add(received_identity)
					except Exception as b:
						print(authenticated_identity)
						print(client_weights_dict)
						print(b)
					break

				global_weights["end"] = 0
				aggregated_response = pickle.dumps(global_weights)

				print(f"Aggregation complete. Sending new global model to: {address}")
				connection.sendall(aggregated_response)

				loss_response = bytearray()
				while True:
					try:
						chunk = connection.recv(1024)
						loss_response.extend(chunk)
						print(chunk)
						break
					except socket.timeout:
						break
				loss_string = loss_response.decode()
				loss_list.append(float(loss_string))
				final_loss = compute_loss()

				response = f"Final loss: {final_loss}"
				connection.sendall(response.encode())

				accuracy_response = bytearray()
				while True:
					try:
						chunk = connection.recv(1024)
						accuracy_response.extend(chunk)
						break
					except socket.timeout:
						break

				accuracy_string = accuracy_response.decode()
				accuracy_list.append(float(accuracy_string))

				final_accuracy = compute_accuracy()

				print(f"Round complete for {address}. Final accuracy: {final_accuracy}")

				end_response_check += 1
				while end_response_check < active_clients_count - awaiting_clients:
					# Wait until all clients are processed so the round condidtions are synchronised among all threads.
					continue
				
				if isSaved:
					# Reset isSaved flag for results
					isSaved = False

				with round_condition:
					# Synchronised FL round completion
					round_finish_count += 1
					if round_finish_count == active_clients_count:
						updateAccuracyTable(final_accuracy)
						updateLossTable(final_loss)
						# Variable reset and adjust
						global_round += 1
						round_finish_count = 0
						client_responses = 0
						all_weights = []
						active_clients_count = clients_connected
						BYZANTINE_CLIENTS = math.ceil(clients_connected/2) - 1
						client_weights_dict.clear()
						loss_list.clear()
						accuracy_list.clear()
						round_condition.notify_all() 
				local_round += 1

			if not isSaved:
				isSaved = True
				save_results()

			connection.close()
	finally:
		print(f"(IBI) Client {authenticated_identity} (address {address}) disconnected.")


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.bind((HOST, PORT))
	s.listen()
	print(f"Now listening on {HOST}:{PORT}")

	while True:
		conn, thread_addr = s.accept()
		print(f"Connection detected: {thread_addr}")
		client_handler = threading.Thread(target=handle_client, args=(conn, thread_addr))
		client_handler.start()