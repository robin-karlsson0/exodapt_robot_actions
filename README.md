# Exodapt AI Robot Actions
Exodapt AI robot actions implemented as ROS 2 action servers. 


# Installation and build

Create and source virtual environment
```
# Create virtual environment with uv
uv venv actions_env
source decidapt_env/bin/activate

# Source ROS 2 after activating virtual environment
source /opt/ros/jazzy/setup.bash
```

Pull package dependencies
```
exodapt_robot_interfaces
exodapt_robot_pt
```
**TODO: Git submodule `exodapt_robot_pt` needs pulling?**

Update git submodules
```bash
git submodule update --remote
```

Install python package dependencies
```
uv pip install -r requirements.txt
```

Build ROS 2 packages
```
cd ros2_ws

colcon build --symlink-install

source install/setup.bash
```


# Implemented actions

## Reply action

### Action servers

`reply_action_server`: `ReplyActionServer` takes a state and optional instruction and returns the generated response.

### Topics

`/reply_action`: Topic where ReplyAction results are published