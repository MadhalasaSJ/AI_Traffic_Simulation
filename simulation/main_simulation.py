import traci
import torch
import time
import numpy as np

# BiLSTM Model Definition
class BiLSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, light_output_size, volume_output_size):
        super(BiLSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, 
            num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.fc_light = torch.nn.Linear(hidden_size * 2, light_output_size)  # For traffic light prediction
        self.fc_volume = torch.nn.Linear(hidden_size * 2, volume_output_size)  # For traffic volume prediction

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Use the last time step
        light_output = torch.sigmoid(self.fc_light(lstm_out))  # Sigmoid for binary classification (light state)
        volume_output = torch.relu(self.fc_volume(lstm_out))  # ReLU for non-negative output (volume prediction)
        return light_output, volume_output

# Function to load the model
def load_model(model_path, input_size, hidden_size, num_layers, light_output_size, volume_output_size):
    model = BiLSTMModel(input_size, hidden_size, num_layers, light_output_size, volume_output_size)
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

# Function to predict traffic conditions using the BiLSTM model
def predict_traffic(model, data):
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)
        light_prediction, volume_prediction = model(data_tensor)
        return light_prediction.item(), volume_prediction.item()

# Function to classify traffic types based on volume
def classify_traffic_type(volume_prediction):
    if volume_prediction < 5:
        return "Low Traffic"
    elif 5 <= volume_prediction < 15:
        return "Medium Traffic"
    else:
        return "High Traffic"

# Function to control traffic lights dynamically based on predictions
def control_traffic_lights_with_model(model, step, light_update_frequency):
    junctions = traci.trafficlight.getIDList()  # Get list of junctions in the network
    for junction in junctions:
        try:
            
            controlled_links = traci.trafficlight.getControlledLinks(junction)
            edge_ids = [link[0][0].split('_')[0] for link in controlled_links]  # Extract edge IDs

            traffic_data = []
            for edge in edge_ids:
                try:
                    vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
                    traffic_data.append(vehicle_count)
                except Exception as e:
                    print(f"Error collecting data for edge '{edge}': {e}")

            
            normalized_data = [x / 20 for x in traffic_data] + [0] * (256 - len(traffic_data))

           
            if step % light_update_frequency == 0:
                light_prediction, volume_prediction = predict_traffic(model, normalized_data)

               
                print(f"Junction {junction}: Light Prediction = {light_prediction:.2f}, Volume Prediction = {volume_prediction:.2f}")

                
                traffic_type = classify_traffic_type(volume_prediction)
                print(f"Junction {junction}: Traffic Type = {traffic_type}")

            
                green_light_duration = 10 
                if volume_prediction > 20:  
                    green_light_duration = 30
                elif volume_prediction > 10:  
                    green_light_duration = 20
                
                
                light_state = "Green" if light_prediction > 0.1 else "Red"
                current_phase = 0 if light_state == "Green" else 1

               
                if light_state == "Green":
                    
                    traci.trafficlight.setPhaseDuration(junction, green_light_duration)
                else:
                    
                    traci.trafficlight.setPhaseDuration(junction, 5)

                # Set the traffic light phase
                traci.trafficlight.setPhase(junction, current_phase)

        except Exception as e:
            print(f"Error controlling traffic lights at junction {junction}: {e}")


def force_exit_remaining_vehicles():
    remaining_vehicles = traci.vehicle.getIDList()
    for vehicle in remaining_vehicles:
        print(f"Removing stuck vehicle: {vehicle}")
        traci.vehicle.remove(vehicle, reason=traci.constants.REMOVE_TELEPORT)

# Initialize SUMO simulation
def initialize_simulation(config_file):
    sumo_binary = "sumo-gui"  
    try:
        traci.start([sumo_binary, "-c", config_file])
        print("SUMO simulation started with GUI.")
    except Exception as e:
        raise RuntimeError(f"Error starting SUMO simulation: {e}")

# Run the simulation
def run_simulation_with_model(simulation_time, model, light_update_frequency):
    try:
        step = 0
        while step < simulation_time:
            traci.simulationStep()  
            control_traffic_lights_with_model(model, step, light_update_frequency)  # Control traffic lights dynamically
            time.sleep(0.01)  
            step += 1
        
        
        force_exit_remaining_vehicles()
    except Exception as e:
        raise RuntimeError(f"Error during simulation: {e}")
    finally:
        print("Simulation completed.")


def stop_simulation():
    try:
        traci.close()
        print("SUMO simulation stopped.")
    except Exception as e:
        print(f"Error stopping SUMO simulation: {e}")

# Main function
def main():
    config_file ="your_simulation.sumocfg"
    model_path = "bilstm_traffic_model.pth"
    input_size = 256
    hidden_size = 128
    num_layers = 2
    light_output_size = 1
    volume_output_size = 1
    simulation_time = 600 
    light_update_frequency = 5  

    try:
        # Load the BiLSTM model
        model = load_model(model_path, input_size, hidden_size, num_layers, light_output_size, volume_output_size)
        
        # Initialize the SUMO simulation
        initialize_simulation(config_file)
        
        # Run the simulation with the model
        run_simulation_with_model(simulation_time, model, light_update_frequency)
    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
       
        stop_simulation()

if __name__ == "__main__":
    main()
