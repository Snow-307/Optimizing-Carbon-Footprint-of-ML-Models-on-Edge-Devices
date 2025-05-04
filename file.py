import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants and configuration parameters
SIMULATION_TIME = 24 * 60 * 60  # 24 hours in seconds
RANDOM_SEED = 42

# Create simulation environment
random.seed(RANDOM_SEED)
env = simpy.Environment()

class NetworkProtocol:
    def __init__(self, protocol_type):
        self.protocol_type = protocol_type  # 'wifi', 'mqtt', 'lora'
        
        # Set protocol-specific parameters
        if protocol_type == 'wifi':
            self.power_consumption = 0.7  # Watts during transmission
            self.transmission_rate = 54000  # KB/s (54 Mbps)
            self.setup_time = 0.1  # seconds to establish connection
        elif protocol_type == 'mqtt':
            self.power_consumption = 0.5  # Watts
            self.transmission_rate = 10000  # KB/s
            self.setup_time = 0.2  # seconds
        elif protocol_type == 'lora':
            self.power_consumption = 0.1  # Watts
            self.transmission_rate = 0.3  # KB/s (slow but efficient)
            self.setup_time = 0.5  # seconds
            
    def calculate_transmission_energy(self, data_size_kb, device_power):
        # Calculate time needed for transmission
        transmission_time = data_size_kb / self.transmission_rate + self.setup_time
        
        # Calculate energy consumed
        energy = self.power_consumption * transmission_time
        
        return energy, transmission_time

class EdgeDevice:
    def __init__(self, env, device_id, device_type, model_type, network_protocol):
        self.env = env
        self.id = device_id
        self.device_type = device_type  # 'raspberry_pi', 'esp32', 'arduino'
        self.model_type = model_type  # 'cnn', 'lstm', etc.
        self.network = NetworkProtocol(network_protocol)
        
        # Set device-specific parameters
        if device_type == 'raspberry_pi':
            self.idle_power = 1.4  # Watts
            self.processing_power = 2.7  # Watts during ML inference
            self.memory = 8000  # MB
        elif device_type == 'esp32':
            self.idle_power = 0.15  # Watts
            self.processing_power = 0.5  # Watts during ML inference
            self.memory = 520  # KB
        elif device_type == 'arduino':
            self.idle_power = 0.05  # Watts
            self.processing_power = 0.2  # Watts during ML inference
            self.memory = 32  # KB
            
        # Performance metrics
        self.total_energy = 0  # Total energy consumed in Joules
        self.inferences = 0
        self.data_sent = 0  # Bytes
        self.action = env.process(self.run())
        
    def run(self):
        while True:
            # Wait for next inference event
            yield self.env.timeout(random.expovariate(1/300))  # ~5 minutes average
            
            # Perform inference
            inference_time = self.perform_inference()
            yield self.env.timeout(inference_time)
            
            # Transmit data
            transmission_time, data_size = self.transmit_data()
            yield self.env.timeout(transmission_time)
    
    def perform_inference(self):
        # Calculate inference time based on device and model
        if self.model_type == 'cnn':
            if self.device_type == 'raspberry_pi':
                inference_time = random.uniform(0.1, 0.5)  # seconds
            elif self.device_type == 'esp32':
                inference_time = random.uniform(0.5, 2.0)
            else:  # arduino
                inference_time = random.uniform(2.0, 5.0)
        else:  # other model types
            inference_time = random.uniform(0.2, 1.0)
        
        # Calculate energy consumed during inference
        energy = self.processing_power * inference_time
        self.total_energy += energy
        self.inferences += 1
        
        return inference_time
    
    def transmit_data(self):
        # Generate random data size based on inference result
        if self.model_type == 'cnn':
            data_size_kb = random.uniform(5, 20)  # KB
        else:
            data_size_kb = random.uniform(1, 10)  # KB
        
        # Calculate energy and time for transmission
        energy, transmission_time = self.network.calculate_transmission_energy(
            data_size_kb, self.processing_power)
        
        self.total_energy += energy
        self.data_sent += data_size_kb
        
        return transmission_time, data_size_kb

class CarbonFootprintCalculator:
    def __init__(self, region='global'):
        # Carbon intensity in kg CO2e per kWh for different regions
        self.carbon_intensity = {
            'global': 0.475,
            'us': 0.389,
            'eu': 0.275,
            'china': 0.555,
            'india': 0.718
        }
        self.region = region
        
    def calculate_emissions(self, energy_joules):
        # Convert joules to kWh
        energy_kwh = energy_joules / 3600000
        
        # Calculate emissions in kg CO2e
        emissions_kg = energy_kwh * self.carbon_intensity[self.region]
        
        return emissions_kg

class OptimizedEdgeDevice(EdgeDevice):
    def __init__(self, env, device_id, device_type, model_type, network_protocol, optimization):
        super().__init__(env, device_id, device_type, model_type, network_protocol)
        self.optimization = optimization  # 'quantization', 'pruning', 'sleep_mode'
        
        # Apply optimization effects
        if optimization == 'quantization':
            # Model quantization reduces processing power but may increase inference time
            self.processing_power *= 0.7
        elif optimization == 'pruning':
            # Model pruning reduces processing power and inference time
            self.processing_power *= 0.8
            # Will affect inference time calculation
        elif optimization == 'sleep_mode':
            # Sleep mode dramatically reduces idle power
            self.idle_power *= 0.3
            # Will need to account for wake-up energy cost
            
    def perform_inference(self):
        # Override to account for optimizations
        inference_time = super().perform_inference()
        
        if self.optimization == 'pruning':
            inference_time *= 0.8  # 20% faster inference
        elif self.optimization == 'quantization':
            inference_time *= 1.1  # 10% slower due to less precision
            
        return inference_time
        
    def calculate_idle_energy(self, time_idle):
        # Calculate energy consumed during idle periods
        energy = self.idle_power * time_idle
        self.total_energy += energy
        return energy
    
def run_simulation(env, duration):
    # Create data collectors
    energy_data = defaultdict(list)
    carbon_data = defaultdict(list)
    
    # Create carbon calculator
    carbon_calc = CarbonFootprintCalculator('global')
    
    # Create devices with different configurations
    devices = []
    
    # Raspberry Pi devices
    for i in range(5):
        for protocol in ['wifi', 'mqtt', 'lora']:
            for model in ['cnn', 'lstm']:
                device = EdgeDevice(
                    env, f"rpi_{i}_{protocol}_{model}", 
                    'raspberry_pi', model, protocol
                )
                devices.append(device)
    
    # ESP32 devices
    for i in range(5):
        for protocol in ['wifi', 'mqtt', 'lora']:
            for model in ['cnn', 'lstm']:
                device = EdgeDevice(
                    env, f"esp32_{i}_{protocol}_{model}", 
                    'esp32', model, protocol
                )
                devices.append(device)
    
    # Arduino devices
    for i in range(5):
        for protocol in ['mqtt', 'lora']:  # Arduino might not use WiFi directly
            for model in ['lstm']:  # Arduino might be limited to smaller models
                device = EdgeDevice(
                    env, f"arduino_{i}_{protocol}_{model}", 
                    'arduino', model, protocol
                )
                devices.append(device)
    
    # Record data at regular intervals
    def data_collector():
        while True:
            # Wait for the next collection interval
            yield env.timeout(3600)  # Collect data every hour
            
            # Record energy consumption for each device configuration
            for device in devices:
                config = f"{device.device_type}_{device.network.protocol_type}_{device.model_type}"
                energy_data[config].append(device.total_energy)
                
                # Calculate carbon emissions
                emissions = carbon_calc.calculate_emissions(device.total_energy)
                carbon_data[config].append(emissions)
    
    env.process(data_collector())
    
    # Run the simulation
    env.run(until=duration)
    
    return devices, energy_data, carbon_data

def analyze_results(devices, energy_data, carbon_data):
    # Calculate total energy and carbon by device type
    device_energy = defaultdict(float)
    device_carbon = defaultdict(float)
    
    for device in devices:
        device_energy[device.device_type] += device.total_energy
        device_carbon[device.device_type] += carbon_data[f"{device.device_type}_{device.network.protocol_type}_{device.model_type}"][-1]
    
    # Calculate energy efficiency by network protocol
    protocol_energy_per_byte = defaultdict(list)
    
    for device in devices:
        if device.data_sent > 0:
            efficiency = device.total_energy / device.data_sent
            protocol_energy_per_byte[device.network.protocol_type].append(efficiency)
    
    # Calculate average energy per inference by model type
    model_energy_per_inference = defaultdict(list)
    
    for device in devices:
        if device.inferences > 0:
            energy_per_inference = device.total_energy / device.inferences
            model_energy_per_inference[device.model_type].append(energy_per_inference)
    
    return {
        'device_energy': device_energy,
        'device_carbon': device_carbon,
        'protocol_efficiency': {k: np.mean(v) for k, v in protocol_energy_per_byte.items()},
        'model_efficiency': {k: np.mean(v) for k, v in model_energy_per_inference.items()}
    }

def visualize_results(results):
    # Create plots for different aspects of the analysis
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot device energy consumption
    axs[0, 0].bar(results['device_energy'].keys(), results['device_energy'].values())
    axs[0, 0].set_title('Total Energy Consumption by Device Type')
    axs[0, 0].set_ylabel('Energy (Joules)')
    
    # Plot device carbon footprint
    axs[0, 1].bar(results['device_carbon'].keys(), results['device_carbon'].values())
    axs[0, 1].set_title('Carbon Footprint by Device Type')
    axs[0, 1].set_ylabel('CO2e (kg)')
    
    # Plot network protocol efficiency
    axs[1, 0].bar(results['protocol_efficiency'].keys(), results['protocol_efficiency'].values())
    axs[1, 0].set_title('Energy Efficiency by Network Protocol')
    axs[1, 0].set_ylabel('Joules per KB')
    
    # Plot model efficiency
    axs[1, 1].bar(results['model_efficiency'].keys(), results['model_efficiency'].values())
    axs[1, 1].set_title('Energy per Inference by Model Type')
    axs[1, 1].set_ylabel('Joules per Inference')
    
    plt.tight_layout()
    plt.savefig('carbon_footprint_results.png')
    plt.show()

def compare_optimizations():
    env = simpy.Environment()
    optimizations = ['none', 'quantization', 'pruning', 'sleep_mode']
    
    all_devices = []
    for opt in optimizations:
        devices = []
        # Create devices with different optimizations
        for i in range(3):
            if opt == 'none':
                device = EdgeDevice(env, f"rpi_{i}", 'raspberry_pi', 'cnn', 'wifi')
            else:
                device = OptimizedEdgeDevice(env, f"rpi_{i}_{opt}", 
                                           'raspberry_pi', 'cnn', 'wifi', opt)
            devices.append(device)
        
        all_devices.append((opt, devices))
    
    # Run simulation
    env.run(until=86400)  # 24 hours
    
    # Compare results
    for opt, devices in all_devices:
        avg_energy = sum(d.total_energy for d in devices) / len(devices)
        print(f"Optimization: {opt}, Average Energy: {avg_energy:.2f} Joules")
    
def main():
    # Initialize simulation environment
    env = simpy.Environment()
    
    # Run the simulation
    print("Running simulation...")
    devices, energy_data, carbon_data = run_simulation(env, SIMULATION_TIME)
    
    # Analyze results
    print("Analyzing results...")
    results = analyze_results(devices, energy_data, carbon_data)
    
    # Visualize results
    print("Generating visualizations...")
    visualize_results(results)
    
    # Print summary
    print("\nSummary of Carbon Footprint Analysis:")
    print(f"Total simulation time: {SIMULATION_TIME/3600:.1f} hours")
    print(f"Total devices simulated: {len(devices)}")
    
    # Calculate totals
    total_energy = sum(device.total_energy for device in devices)
    total_carbon = sum(results['device_carbon'].values())
    total_inferences = sum(device.inferences for device in devices)
    
    print(f"Total energy consumed: {total_energy/3600:.2f} Watt-hours")
    print(f"Total carbon footprint: {total_carbon:.6f} kg CO2e")
    print(f"Carbon intensity: {total_carbon/(total_energy/3600000):.4f} kg CO2e per kWh")
    print(f"Total inferences performed: {total_inferences}")
    print(f"Average energy per inference: {total_energy/total_inferences:.4f} Joules")
    
    # Print most efficient configurations
    best_device = min(devices, key=lambda d: d.total_energy/d.inferences if d.inferences > 0 else float('inf'))
    print(f"\nMost energy-efficient configuration:")
    print(f"  Device: {best_device.device_type}")
    print(f"  Model: {best_device.model_type}")
    print(f"  Network: {best_device.network.protocol_type}")
    print(f"  Energy per inference: {best_device.total_energy/best_device.inferences:.4f} Joules")
    
    # Optionally run optimization comparison
    print("\nComparing optimization strategies...")
    compare_optimizations()

if __name__ == "__main__":
    main()