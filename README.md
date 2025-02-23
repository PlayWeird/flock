# Multi-Agent Flocking Simulation

A Python-based simulation of multi-agent flocking behavior with obstacle avoidance and target tracking capabilities. This project implements a distributed control algorithm for autonomous agents that exhibit coordinated movement while avoiding obstacles and following a target.

## Features

- Real-time visualization of flocking behavior
- Multiple movement patterns for target tracking:
  - Static position
  - Circular motion
  - Wave motion
- Dynamic obstacle avoidance
- Adaptive window scaling
- Performance metrics tracking:
  - Agent velocities
  - Network connectivity
- OpenGL-based rendering using Pyglet

## Requirements

```
numpy
pyglet
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flocking-simulation.git
cd flocking-simulation
```

2. Install required packages:
```bash
pip install numpy pyglet matplotlib
```

## Usage

Run the simulation:
```bash
python main.py
```

### Configuration

The simulation can be configured by modifying the following parameters in the `SimWindow` class:

- `number_of_sensor_nodes`: Number of agents in the simulation (default: 7)
- `run_interval`: Duration of simulation in frames (default: 10000)
- `neighbor_distance`: Desired distance between agents (default: 15.0)
- `sensor_range`: Range at which agents can detect each other (default: 1.2 × neighbor_distance)
- `obstacle_flag`: Enable/disable obstacles (default: False)
- `target_path`: Movement pattern for the target ('', 'circle', or 'wave')

## How It Works

### Agent Behavior

Each agent (sensor node) follows three main behavioral rules:

1. **Flocking Behavior**:
   - Cohesion: Agents try to stay close to their neighbors
   - Alignment: Agents try to match velocity with neighbors
   - Separation: Agents avoid getting too close to each other

2. **Obstacle Avoidance**:
   - Agents detect obstacles within their sensor range
   - Virtual nodes (beta agents) are created for obstacle avoidance calculations
   - Smooth avoidance maneuvers using potential field methods

3. **Target Tracking**:
   - Agents collectively follow a target
   - Target can move in different patterns (static, circular, wave)
   - Balanced with flocking and obstacle avoidance behaviors

### Visualization

The simulation provides real-time visualization using Pyglet and OpenGL:
- Agents are displayed as filled circles
- Sensor ranges are shown as transparent circles
- Obstacles are rendered as concentric circles
- Agent connections are shown as lines between neighbors
- Window automatically scales to keep all agents in view

### Performance Metrics

The simulation generates two plots after completion:
1. Individual agent velocities over time
2. Network connectivity metric over time

## Technical Details

### Key Classes

- `SimWindow`: Main simulation window and controller
- `SensorNode`: Individual agent implementation
- `CircularObstacle`: Obstacle representation
- `TargetNode`: Target node implementation
- `BetaNode`: Virtual node for obstacle avoidance

### Mathematical Framework

The simulation uses a distributed control algorithm based on:
- Smooth potential functions for inter-agent interactions
- Gradient-based control for flocking behavior
- Velocity consensus for alignment
- Beta-agent approach for obstacle avoidance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on flocking behavior research and distributed control algorithms for multi-agent systems. Special thanks to contributors to the Pyglet and NumPy libraries that made this visualization possible.
