import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import csv


# Constants
SPEED_OF_SOUND = 343.0  # m/s
MIC_DISTANCES = [0.055, 0.063, 0.055]  # distance between mics in each array


def save_to_csv(filename, h, k, theta, d):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # First row: source position
        writer.writerow([h, k, theta, d])
            

def time_delay_to_angle(delta_t, mic_index):
    """Convert time delay to angle of arrival"""
    d = MIC_DISTANCES[mic_index]
    sin_theta = (SPEED_OF_SOUND * delta_t) / d
    # Clamp to valid range to avoid numerical issues
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    return np.arcsin(sin_theta)

def angle_difference(angle1, angle2):
    """Compute the smallest difference between two angles (handles wraparound)"""
    diff = angle1 - angle2
    # Normalize to [-pi, pi]
    return np.arctan2(np.sin(diff), np.cos(diff))

def compute_expected_angle(mic_pose, source_pos):
    """
    Compute expected angle from mic array to source
    
    mic_pose: [x_mic, y_mic, theta_mic] where theta_mic is array orientation
    source_pos: [x_src, y_src]
    
    Returns: angle in array's local frame
    """
    x_mic, y_mic, theta_mic = mic_pose
    x_src, y_src = source_pos
    
    # Angle from mic to source in global frame
    angle_global = np.arctan2(y_src - y_mic, x_src - x_mic)
    
    # Convert to array's local frame
    angle_local = angle_global - theta_mic
    
    return angle_local

def objective_function(mic_pose, measured_angles, source_positions):
    """
    Objective function: sum of squared angle errors
    
    mic_pose: [x_mic, y_mic, theta_mic]
    measured_angles: list of measured angles from array
    source_positions: list of [x, y] source positions
    """
    total_error = 0.0
    
    for measured_angle, source_pos in zip(measured_angles, source_positions):
        expected_angle = compute_expected_angle(mic_pose, source_pos)
        error = angle_difference(measured_angle, expected_angle)
        total_error += error**2
    
    return total_error

def load_and_process_data(mic_index, position_index):
    """Load data and compute mean angle for one position"""
    filename = f'data_{mic_index+1}_{position_index+1}.csv'
    df = pd.read_csv(filename, header=None)
    
    # First row contains source position
    x_src, y_src = df.iloc[0].values
    
    # Remaining rows contain time delays
    time_delays = df.iloc[1:, 0].values
    mean_time_delay = np.mean(time_delays)
    
    # Convert to angle
    angle = time_delay_to_angle(mean_time_delay, mic_index)
    
    return (x_src, y_src), angle

def calibrate_microphone_array(mic_index, num_positions, method='nelder-mead', 
                               initial_guess=None, use_global=False):
    """
    Calibrate microphone array position and orientation
    
    mic_index: which microphone array (0, 1, or 2)
    num_positions: number of calibration positions
    method: optimization method ('nelder-mead', 'powell', 'bfgs')
    initial_guess: [x, y, theta] initial guess (if None, uses [0, 0, 0])
    use_global: if True, use global optimization (slower but more robust)
    """
    # Load all calibration data
    source_positions = []
    measured_angles = []
    
    for pos_idx in range(num_positions):
        source_pos, angle = load_and_process_data(mic_index, pos_idx)
        source_positions.append(source_pos)
        measured_angles.append(angle)
    
    print(f"\nCalibrating Microphone Array {mic_index + 1}")
    print(f"Number of calibration points: {num_positions}")
    print(f"Source positions: {source_positions}")
    print(f"Measured angles (deg): {[np.degrees(a) for a in measured_angles]}")
    
    # Initial guess
    if initial_guess is None:
        initial_guess = [5.0, 5.0, 5.3]
    
    if use_global:
        # Global optimization - more robust but slower
        # Define bounds: assume mics are within a reasonable area
        bounds = [(-5, 5), (-5, 5), (-np.pi, np.pi)]
        result = differential_evolution(
            objective_function,
            bounds,
            args=(measured_angles, source_positions),
            seed=42,
            atol=1e-6,
            tol=1e-6
        )
    else:
        # Local optimization
        result = minimize(
            objective_function,
            initial_guess,
            args=(measured_angles, source_positions),
            method=method,
            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
    
    if result.success:
        x_mic, y_mic, theta_mic = result.x
        print(f"\n✓ Optimization successful!")
        print(f"Microphone position: ({x_mic:.4f}, {y_mic:.4f})")
        print(f"Microphone orientation: {np.degrees(theta_mic):.2f}°")
        print(f"Final error (sum of squared angle errors): {result.fun:.6f}")
        print(f"RMS angle error: {np.sqrt(result.fun / num_positions):.4f} rad "
              f"({np.degrees(np.sqrt(result.fun / num_positions)):.2f}°)")

        save_to_csv(f'sensor{mic_index+1}.csv', x_mic, y_mic, theta_mic, MIC_DISTANCES[mic_index])

    else:
        print(f"\n✗ Optimization failed: {result.message}")
    
    return result.x, result.fun

def visualize_calibration(mic_pose, source_positions, measured_angles):
    """Visualize the calibration result"""
    x_mic, y_mic, theta_mic = mic_pose
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot source positions
    src_x = [pos[0] for pos in source_positions]
    src_y = [pos[1] for pos in source_positions]
    ax.scatter(src_x, src_y, c='red', s=100, marker='o', label='Sources', zorder=3)
    
    # Plot microphone
    ax.scatter([x_mic], [y_mic], c='blue', s=200, marker='^', label='Microphone', zorder=3)
    
    # Draw microphone orientation
    arrow_len = 0.5
    ax.arrow(x_mic, y_mic, arrow_len * np.cos(theta_mic), arrow_len * np.sin(theta_mic),
             head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)
    
    # Draw lines from mic to sources
    for i, (source_pos, measured_angle) in enumerate(zip(source_positions, measured_angles)):
        ax.plot([x_mic, source_pos[0]], [y_mic, source_pos[1]], 
                'g--', alpha=0.5, linewidth=1)
        
        # Compute and display angle error
        expected_angle = compute_expected_angle(mic_pose, source_pos)
        error = angle_difference(measured_angle, expected_angle)
        
        # Add text annotation
        mid_x = (x_mic + source_pos[0]) / 2
        mid_y = (y_mic + source_pos[1]) / 2
        ax.text(mid_x, mid_y, f'{np.degrees(error):.1f}°', fontsize=8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Microphone Array Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('calibration_result.png', dpi=150)
    plt.show()

# Main execution
if __name__ == "__main__":
     # Which microphone array (0, 1, or 2)
    num_positions = 7  # Number of calibration positions



    for mic_index in range(0, 3):
        # Try with local optimization first
        result, error = calibrate_microphone_array(
            mic_index=mic_index,
            num_positions=num_positions,
            method='nelder-mead',
            initial_guess=[0.0, 0.0, 0.0],
            use_global=False
        )
        
        # If error is too high, try global optimization
        rms_error_deg = np.degrees(np.sqrt(error / num_positions))
        if rms_error_deg > 10:  # If RMS error > 10 degrees
            print("\n⚠ Error too high, trying global optimization...")
            result, error = calibrate_microphone_array(
                mic_index=mic_index,
                num_positions=num_positions,
                use_global=True
            )
            
            # Load data for visualization
        source_positions = []
        measured_angles = []
        for pos_idx in range(num_positions):
            source_pos, angle = load_and_process_data(mic_index, pos_idx)
            source_positions.append(source_pos)
            measured_angles.append(angle)
                
                # Visualize
        visualize_calibration(result, source_positions, measured_angles)
