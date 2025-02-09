from roboticstoolbox import DHRobot, RevoluteDH

class SymbolicRobot(DHRobot):
    def __init__(self, a1, alpha1, d1):
        # Define the robot links using RevoluteDH
        links = [
            RevoluteDH(a=a1, alpha=alpha1, d=d1),  # Do not pass theta here
            # Add more links if necessary
        ]
        
        # Initialize the robot using the base class DHRobot
        super().__init__(links, name="SymbolicRobot")

# Now, when creating an instance, pass the required parameters
a1, alpha1, d1 = 1.0, 0.0, 0.0
robot = SymbolicRobot(a1, alpha1, d1)

# Print the robot details
print(robot)
