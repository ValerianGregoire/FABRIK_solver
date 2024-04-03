import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt

class Leg:

    def __init__(self,
                    joints:int = 3,
                    length:list = 1,
                    origin_coordinates:list = np.array([0,0,0]),
                    start_direction:list = np.array([1,0,0]),
                    joints_limits:list = []):
        """
        joints: Number of joints, including the origin and end-point.
        length: Numpy array containing the length of each leg segment.
        origin_coordinates: Numpy array with the x,y,z coordinates of the first
        joint. The coordinates must be in a left-hand basis.
        start_direction: Normalized numpy array pointing towards the direction 
        of the leg at start.
        joints_limits: [joints x 3 x 2] matrix containing lower and upper angle
        limits for each joint. Optionnal*        
        """
        
        # Default values - TO READ ONLY
        self._p = [origin_coordinates] # Joints coordinates
        try:
            for idx in range(joints-1):
                self._p.append(self._p[-1] + length[idx]*start_direction)
            self._d = np.copy(length) # Segments distances
        except TypeError:
            for idx in range(joints-1):
                self._p.append(self._p[-1] + length*start_direction)
            self._d = np.array([length for i in range(joints)]) # Segments distances
            

        self._a = [np.zeros(3) for idx in range(len(self._p))] # Angles between joints
        self._limits = np.copy(joints_limits) # Joints angle limits
        self._t = np.copy(self._p[-1])

        # Active values - TO WRITE
        self.p = np.copy(self._p)
        self.a = np.copy(self._a)
        self.d = np.copy(self._d)
        self.limits = np.copy(self._limits)
        self.t = np.copy(self._t)

        # Additional parameters
        self.tolerance = 1e-2

    def set_target(self,target:list):
        """
        Updates the target point of the leg with added noise for mathematical
        errors correction.
        """
        self.t = target + np.random.normal(0,0.001,len(target))

    def compute_angles(self, p1, p2, a2):
        """
        Computes the angle in degrees of each joint around all axes.

        p1: Joint idx coordinates
        p2: Joint idx-1 coordinates
        a2 : Joint idx-1 angle
        """
        return np.array([   np.arctan2(p1[2] - p2[2], p1[1] - p2[1]),
                            np.arctan2(p1[2] - p2[2], p1[0] - p2[0]),
                            np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
                            ]) * 180/np.pi - a2[-1]
    
    def show(self, ax) -> None:
        """
        Adds the leg data to a 3d projected plot for visualization.
        """
        # Fetch of points coordinates
        x = [self.p[idx][0] for idx in range(len(self.p))]
        y = [self.p[idx][1] for idx in range(len(self.p))]
        z = [self.p[idx][2] for idx in range(len(self.p))]

        # Plotting of the points
        ax.scatter(x[1:-1],y[1:-1],z[1:-1])
        ax.scatter(x[0],y[0],z[0], color = 'green')
        ax.scatter(x[-1],y[-1],z[-1], color = 'red')
        ax.plot(x,y,z)
            
    def angle_correction(self, idx):
        # Restrict rotation around the x axis
        if self.a[idx][0] > self.limits[idx][0][1] or self.a[idx][0] < self.limits[idx][0][0]:
            if not self.a[idx][0] == 0:
                sign = abs(self.a[idx][0])/self.a[idx][0]
            else:
                sign = 1
            self.p[idx][2] = sign * \
                np.tan(np.radians(self.limits[idx][0][1]))*self.p[idx][1]
        # Restrict rotation around the y axis
        if self.a[idx][1] > self.limits[idx][1][1] or self.a[idx][1] < self.limits[idx][1][0]:
            if not self.a[idx][1] == 0:
                sign = abs(self.a[idx][1])/self.a[idx][1]
            else:
                sign = 1
            self.p[idx][0] = sign * \
                np.tan(np.radians(self.limits[idx][1][1]))*self.p[idx][2]
        # Restrict rotation around the z axis
        if self.a[idx][2] > self.limits[idx][2][1] or self.a[idx][2] < self.limits[idx][2][0]:
            if not self.a[idx][2] == 0:
                sign = abs(self.a[idx][2])/self.a[idx][2]
            else:
                sign = 1
            self.p[idx][1] = sign * \
                np.tan(np.radians(self.limits[idx][2][1]))*self.p[idx][0]
    
    def update(self):
        """
        Applies the FABRIK algorithm to the joints, taking angle limits into 
        account.
        """

        # The target cannot be reached
        if (nl.norm(self.t) - nl.norm(self.p[0])) > sum(self.d):
            return False

        # Target is reachable
        else:
            # Keep the position of the first joint for later backward reaching
            b = np.copy(self.p[0])

            it = 0 # Iteration counter

            # FABRIK iteration
            while (nl.norm(self.p[-1] - self.t) > self.tolerance) and it < 30:
                
                # Set the position of the last joint for forward reach
                self.p[-1] = self.t

                # Forward reach
                for idx in reversed(range(len(self.p)-1)):
                    r = nl.norm(self.p[idx+1] - self.p[idx])
                    l = self.d[idx]/r
                    self.p[idx] = (1-l)*self.p[idx+1] + l*self.p[idx]

                # Backward reach
                self.p[0] = np.copy(b)
                for idx in range(len(self.p)-1):
                    r = nl.norm(self.p[idx+1] - self.p[idx])
                    l = self.d[idx]/r
                    self.p[idx+1] = (1-l)*self.p[idx] + l*self.p[idx+1]
                    if idx > 0:
                        self.a[idx] = self.compute_angles(self.p[idx],
                                                        self.p[idx-1],
                                                        self.a[idx-1])
                    if len(self.limits) > 0:
                        self.angle_correction(idx)
        

                # Iteration counter ++
                it += 1


if __name__ == '__main__':

    # Declaration of leg parameters
    joints = 4
    length = 3
    origin_coordinates = np.array([0,0,0])
    start_direction = np.array([1,0,0])
    joints_limits = np.array([
        [[0, 0], [0, 0], [-90, 90]],
        [[-30, 30], [-30, 30], [0, 0]],
        [[-30, 30], [-30, 30], [0, 0]]
        ])

    # Instantiation of the leg
    leg = Leg(joints=joints, length=length,
               origin_coordinates=origin_coordinates,
               start_direction=start_direction,
               joints_limits=joints_limits)

    # Display of the leg's initial parameters
    print(f"""------------------------------------- Initial parameters

Joints coordinates : {leg.p}

Joints angles : {leg.a}

Joints lenghts : {leg.d}

Joints angular limits : {leg.limits}

""")

    leg.set_target(np.array([8,2,0]))
    leg.update()

    print(f"""------------------------------------- Final parameters

Joints coordinates : {leg.p}

Joints angles : {leg.a}

Joints lenghts : {leg.d}

Joints angular limits : {leg.limits}

""")

    # Display of the leg's position
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_zlim([0,10])
    leg.show(ax)
    plt.show()
