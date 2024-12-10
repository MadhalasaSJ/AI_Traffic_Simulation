import os
import traci
import sumolib

# Define the path to your SUMO installation
SUMO_HOME = os.environ.get('SUMO_HOME', r'')

if not SUMO_HOME:
    print("Please define the SUMO_HOME environment variable to point to your SUMO installation.")
    exit(1)

sumo_config_file = "your_simulation.sumocfg"
sumo_network_file = "network.net.xml"

# Load the network file using sumolib
def load_network():
    net = sumolib.net.readNet(sumo_network_file)
    print("Network loaded successfully.")
    edges = net.getEdges()
    print("Available edges:", [e.getID() for e in edges])
    return net

# Start the SUMO simulation with the GUI
def start_sumo_simulation():
    sumo_cmd = [os.path.join(SUMO_HOME, "bin", "sumo-gui"), "-c", sumo_config_file]
    try:
        traci.start(sumo_cmd)
        print("Simulation started with SUMO-GUI.")
    except Exception as e:
        print(f"Error starting SUMO simulation: {e}")
        exit(1)

# Add vehicles to the simulation
def add_vehicles():
    for i in range(10):
        vehicle_id = f"vehicle_{i}"
        route = "route1"  # Ensure this route exists in the .rou.xml file
        depart_time = 0
        vehicle_type = "car"
        traci.vehicle.add(vehicle_id, route, typeID=vehicle_type, depart=depart_time)
        traci.vehicle.setTau(vehicle_id, 2.0)  # Set tau to 2.0 for each vehicle
        print(f"Vehicle {vehicle_id} added to route {route}.")

# Run the simulation steps
def run_simulation():
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        for vehicle_id in traci.vehicle.getIDList():
            position = traci.vehicle.getPosition(vehicle_id)
            print(f"Vehicle {vehicle_id} at position {position}")
    print("Simulation ended.")

# Main function
def main():
    try:
        load_network()
        start_sumo_simulation()
        add_vehicles()
        run_simulation()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if traci.simulation.getMinExpectedNumber() == 0:
            traci.close()

if __name__ == "__main__":
    main()
