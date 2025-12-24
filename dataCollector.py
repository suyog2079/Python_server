import socket
import threading
import time
import csv
import numpy as np
from collections import defaultdict
from datetime import datetime

class CalibrationServer:
    def __init__(self, host='192.168.12.171', port=6060, num_clients=3):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        
        self.clients = []
        self.client_data = defaultdict(list)  # {client_id: [data_lines]}
        self.client_status = {}  # {client_id: 'connected'/'disconnected'}
        self.clients_ready = threading.Event()
        self.recording = False
        self.lock = threading.Lock()
        
        self.position_index = 0  # Which calibration position we're on
        
    def handle_client(self, client_socket, client_address, client_id):
        """Handle individual client connection"""
        print(f"[Client {client_id}] Connected from {client_address}")
        
        with self.lock:
            self.client_status[client_id] = 'connected'
        
        try:
            buffer = ""
            last_data_time = time.time()
            
            while True:
                client_socket.settimeout(5.0)  # 5 second timeout
                
                try:
                    data = client_socket.recv(1024).decode('utf-8')
                except socket.timeout:
                    # Check if client is still alive
                    if time.time() - last_data_time > 10:
                        print(f"[Client {client_id}] Timeout - no data received")
                        break
                    continue
                
                if not data:
                    break
                
                last_data_time = time.time()
                buffer += data
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    
                    if self.recording:
                        with self.lock:
                            self.client_data[client_id].append(line)
                            
        except Exception as e:
            print(f"[Client {client_id}] Error: {e}")
        finally:
            with self.lock:
                self.client_status[client_id] = 'disconnected'
            client_socket.close()
            print(f"[Client {client_id}] Disconnected")
    
    def wait_for_clients(self):
        """Wait for all clients to connect"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(self.num_clients)
            print(f"Server listening on {self.host}:{self.port}")
            print(f"Waiting for {self.num_clients} clients to connect...")
            
            for i in range(self.num_clients):
                client_socket, client_address = server_socket.accept()
                client_id = i + 1
                
                # Start thread to handle this client
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, client_address, client_id)
                )
                client_thread.daemon = True
                client_thread.start()
                
                self.clients.append((client_socket, client_thread))
                print(f"✓ Client {client_id} connected ({i+1}/{self.num_clients})")
            
            print(f"\n✓ All {self.num_clients} clients connected!")
            self.clients_ready.set()
            
        except Exception as e:
            print(f"Server error: {e}")
            server_socket.close()
            raise
        
        return server_socket
    
    def check_clients_status(self):
        """Check if all clients are still connected"""
        with self.lock:
            disconnected = [cid for cid, status in self.client_status.items() 
                          if status == 'disconnected']
        
        if disconnected:
            print(f"\n⚠ Warning: Clients {disconnected} are disconnected!")
            return False
        return True
    
    def record_calibration_point(self, x_source, y_source, duration=2.0):
        """Record data from all clients for one calibration point"""
        
        # Check client status
        if not self.check_clients_status():
            response = input("Some clients are disconnected. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        # Clear previous data
        with self.lock:
            self.client_data.clear()
        
        print(f"\n{'='*60}")
        print(f"Recording calibration point {self.position_index + 1}")
        print(f"Source position: ({x_source}, {y_source})")
        print(f"Recording for {duration} seconds...")
        print(f"{'='*60}")
        
        # Countdown
        print("Starting in: ", end='', flush=True)
        for i in range(3, 0, -1):
            print(f"{i}...", end='', flush=True)
            time.sleep(1)
        print("Recording!")
        
        # Start recording
        self.recording = True
        
        # Show progress
        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            print(f"\rRecording... {remaining:.1f}s remaining", end='', flush=True)
            time.sleep(0.1)
        
        self.recording = False
        print("\n\nRecording complete!")
        
        # Analyze and save data for each client
        with self.lock:
            all_saved = True
            for client_id in range(1, self.num_clients + 1):
                if client_id not in self.client_data:
                    print(f"  ⚠ Client {client_id}: No data received")
                    all_saved = False
                    continue
                
                data_lines = self.client_data[client_id]
                print(f"  Client {client_id}: {len(data_lines)} samples recorded")
                
                # Parse and validate data
                del_t_values = []
                for line in data_lines:
                    try:
                        parts = line.strip().split(',')
                        if len(parts) >= 6:
                            del_t = float(parts[5])
                            print(del_t);
                            del_t_values.append(del_t)
                    except (ValueError, IndexError):
                        continue
                
                if del_t_values:
                    del_t_array = np.array(del_t_values)
                    print(f"    Valid samples: {len(del_t_values)}")
                    print(f"    del_t mean: {np.mean(del_t_array):.6f}s")
                    print(f"    del_t std:  {np.std(del_t_array):.6f}s")
                    
                    # Save to CSV file
                    filename = f'data_{client_id}_{self.position_index + 1}.csv'
                    self.save_to_csv(filename, x_source, y_source, data_lines)
                    print(f"    ✓ Saved to {filename}")
                else:
                    print(f"    ⚠ No valid data to save")
                    all_saved = False
        
        if all_saved:
            self.position_index += 1
            return True
        else:
            print("\n⚠ Some clients had no valid data. Point not counted.")
            return False
    
    def save_to_csv(self, filename, x_source, y_source, data_lines):
        """
        Save calibration data to CSV in the format expected by optimizer:
        - First row: x_source, y_source
        - Remaining rows: del_t values (one per row)
        
        Input data format from client: timestamp,h,k,phi,del_t
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # First row: source position
            writer.writerow([x_source, y_source])
            
            # Extract del_t from each line and write
            for line in data_lines:
                try:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        del_t = float(parts[5])  # del_t is the 5th field (index 4)
                        writer.writerow([del_t])
                except (ValueError, IndexError) as e:
                    continue
    
    def print_summary(self):
        """Print summary of calibration session"""
        print("\n" + "="*60)
        print("CALIBRATION SESSION SUMMARY")
        print("="*60)
        print(f"Total calibration points recorded: {self.position_index}")
        print(f"\nFiles created:")
        for pos_idx in range(self.position_index):
            print(f"\n  Position {pos_idx + 1}:")
            for client_id in range(1, self.num_clients + 1):
                filename = f'data_{client_id}_{pos_idx + 1}.csv'
                print(f"    - {filename}")
        
        print("\n" + "="*60)
        print("Next steps:")
        print("  1. Run the optimizer with mic_index=0, 1, 2")
        print("  2. Each optimizer run will use data_X_Y.csv files")
        print("     where X is the mic index (1-3) and Y is position (1-N)")
        print("="*60)
    
    def run(self):
        """Main server loop"""
        print("="*60)
        print("MICROPHONE ARRAY CALIBRATION SERVER")
        print("="*60)
        
        # Start server and wait for clients
        server_socket = self.wait_for_clients()
        self.clients_ready.wait()
        
        print("\n" + "="*60)
        print("CALIBRATION DATA COLLECTION")
        print("="*60)
        print("\nInstructions:")
        print("  1. Place sound source at a known position")
        print("  2. Enter the X and Y coordinates")
        print("  3. Server will record for specified duration")
        print("  4. Repeat for at least 3 different positions")
        print("="*60)
        
        try:
            while True:
                print(f"\n--- Calibration Point {self.position_index + 1} ---")
                
                # Get source position from user
                try:
                    x_input = input("\nEnter X coordinate of sound source (or 'q' to quit): ").strip()
                    if x_input.lower() == 'q':
                        break
                    x_source = float(x_input)
                    
                    y_input = input("Enter Y coordinate of sound source: ").strip()
                    y_source = float(y_input)
                    
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
                    continue
                
                # Optional: recording duration
                duration_input = input("Recording duration in seconds (default 2.0): ").strip()
                try:
                    duration = float(duration_input) if duration_input else 2.0
                except ValueError:
                    print("Invalid duration, using default 2.0s")
                    duration = 2.0
                
                # Record data
                success = self.record_calibration_point(x_source, y_source, duration)
                
                if success:
                    print(f"\n✓ Calibration point {self.position_index} complete!")
                
                # Ask if user wants to continue
                if self.position_index >= 3:
                    print(f"\nYou have {self.position_index} calibration points.")
                    cont = input("Record another point? (y/n): ").strip().lower()
                    if cont != 'y':
                        break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.print_summary()
            server_socket.close()


def main():
    # Configuration
    HOST = '192.168.12.171'  # Server IP - change this to your server's IP
    PORT = 6060              # Server port
    NUM_CLIENTS = 3          # Number of microphone clients
    
    print("\nConfiguration:")
    print(f"  Host: {HOST}")
    print(f"  Port: {PORT}")
    print(f"  Expected clients: {NUM_CLIENTS}")
    print(f"\nMake sure your Rust clients connect to {HOST}:{PORT}\n")
    
    # Create and run server
    server = CalibrationServer(host=HOST, port=PORT, num_clients=NUM_CLIENTS)
    server.run()


if __name__ == "__main__":
    main()
