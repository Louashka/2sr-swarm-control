import math

# Velocities
OMNI_SPEED = 0.12
ROTATION_SPEED = 1
LU_SPEED = 0.12

# VSF parameters
L_VSS = 75.875 * 10**(-3)  # VSS length
L_VSS_CNN = 30 * 10**(-3)
D_BRIDGE = 7 * 10**(-3)  # bridge width
L_VSF = 2 * L_VSS  # VSF length

# LU parameters
LU_SIDE = 42 * 10**(-3)  # block side length
LU_R = LU_SIDE * math.sqrt(2) / 2 # Distance between LU center and its corner
LU_ALPHA = math.radians(-135) # Angle between LU orientation and r

# Wheels parameters
WHEEL_R = 10 * 10**(-3)  # wheel radius
WHEEL_TH = 12 * 10**(-3)  # wheel thickness
WHEEL_MARGIN = 10**(-3)

BETA = [math.pi / 2, math.pi, -math.pi / 2, math.pi]

H1 = L_VSS_CNN + LU_SIDE - WHEEL_R
H2 = LU_SIDE + WHEEL_TH / 2 + WHEEL_MARGIN
H3 = L_VSS_CNN - WHEEL_TH / 2 - WHEEL_MARGIN
H4 = WHEEL_R

# Wheels coordinates w.r.t. to VSB end frames {b_j}
bj_Q_w = [[-H1, -H3, H3, H1],
          [-H2, -H4, -H4, -H2]]


# Constants of logarithmic spirals

SPIRAL_COEF = [[2.3250 * L_VSS, 3.3041 * L_VSS,
                2.4471 * L_VSS], [0.3165, 0.083, 0.2229]]

SPIRAL_CENTRE = [-0.1223 * L_VSS, 0.1782 * L_VSS]

M = [3 / 2, 1, 3 / 4]

# Motive tracling data
M_POS = ['marker_x', 'marker_y', 'marker_z']
RB_POS = ['x', 'y', 'z']
RB_PARAMS = ['a', 'b', 'c', 'd']
RB_ANGLES = ['roll', 'pitch', 'yaw']

# Coords of the real LU center w.r.t. the rb position
LU_HEAD_CENTER_R = 0.01074968
LU_HEAD_CENTER_ANGLE = math.radians(-60.2551187)