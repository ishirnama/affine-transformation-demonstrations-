import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#reccomeded step L:{ 0.19 < h < 1 }
fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")


def cartesian(v):
    x = v[0][0]
    y = v[1][0]
    z = v[2][0]
    return [x,y,z]


def square(v1,v2,v3,v4,u1,u2,u3,u4, col):
    vertices = [
        v1,v2,v3,v4,  # Bottom face
        u1,u2,u3,u4   # Top face
    ]

# Define the edges that connect the vertices to form the faces of the cube
    edges = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Side face
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Side face
        [vertices[4], vertices[7], vertices[3], vertices[0]]   # Side face
    ]
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 3])
    ax.set_zlim([0, 3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

# Plot the cube
    return ax.add_collection3d(Poly3DCollection(edges, facecolors=col, linewidths=1, edgecolors='r', alpha=.25))

def magnitude(vector):
    return np.linalg.norm(vector)

def no_homo(vector):
    return vector[:3]  # Non-homogeneous vector (first 3 components)

def graph(x0, y0, z0, v, col):
    ax.quiver(x0,y0,z0, v[0][0], v[1][0], v[2][0], arrow_length_ratio=0.05, color=col)
    plt.title("Demo")
    
def vector(x,y,z,w):
    v = np.array([[x],[y],[z],[w]])
    return v

def rotation_matrix_X(phi):
    mat = np.array([
        [1,0,0,0],
        [0, np.cos(phi), -1*np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0,0,0,1]
    ])
    return mat

def scale_matrix(Sx,Sy,Sz,w):
    mat = np.array([
        [Sx,0,0,0],
        [0,Sy,0,0],
        [0,0,Sz,0],
        [0,0,0,w]
    ])
    return mat

def sheer_matrix(dx,dy,w):
    mat = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [dx,dy,1,0],
        [0,0,0,w]
    ])
    return mat

def transform_square(mat):
    h1 = np.matmul(mat, v1)
    h2 = np.matmul(mat, v2)
    h3 = np.matmul(mat, v3)
    h4 = np.matmul(mat, v4)
    g1 = np.matmul(mat, u1)
    g2 = np.matmul(mat, u2)
    g3 = np.matmul(mat, u3)
    g4 = np.matmul(mat, u4)
    print(f"v1' : {h1}")
    print(f"v2' : {h2}")
    print(f"v3' : {h3}")
    print(f"v4' : {h4}")
    print(f"u1' : {g1}")
    print(f"u2' : {g2}")
    print(f"u3' : {g3}")
    print(f"u4' : {g4}")
    square(
        cartesian(h1),
        cartesian(h2),
        cartesian(h3),
        cartesian(h4),
        cartesian(g1),
        cartesian(g2),
        cartesian(g3),
        cartesian(g4),
        'red'
        )
def rotate_square(phi):
    rot = rotation_matrix_X(phi)
    h1 = np.matmul(rot, v1)
    h2 = np.matmul(rot, v2)
    h3 = np.matmul(rot, v3)
    h4 = np.matmul(rot, v4)
    g1 = np.matmul(rot, u1)
    g2 = np.matmul(rot, u2)
    g3 = np.matmul(rot, u3)
    g4 = np.matmul(rot, u4)
    print(f"v1' : {h1}")
    print(f"v2' : {h2}")
    print(f"v3' : {h3}")
    print(f"v4' : {h4}")
    print(f"u1' : {g1}")
    print(f"u2' : {g2}")
    print(f"u3' : {g3}")
    print(f"u4' : {g4}")

    square(
        cartesian(h1),
        cartesian(h2),
        cartesian(h3),
        cartesian(h4),
        cartesian(g1),
        cartesian(g2),
        cartesian(g3),
        cartesian(g4),
        'red'
        )
    

def spawn_sphere(radius):
    # Make data
    r = radius
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)

    # Set an equal aspect ratio
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
def spawn_vector_field(step_length, direction, scale_matrix, x0, y0, z0, domain):
    start = domain[0]
    end = domain[1]
    ray = np.matmul(scale_matrix, direction)
    
    for z0 in np.arange(start, end + step_length, step_length):
        for x0 in np.arange(start, end + step_length, step_length):
            graph(x0, y0, z0, ray, 'red')

def draw_sphere_on_hit(Q, d, radius, step_length, domain):
    start = domain[0]
    end = domain[1]
    # Initialize the 2D grid
    grid_size = int((radius*2) / step_length) + 1  # Considering the range from -4 to 4
    canvas = [[" " for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Iterate over the x and y values
    for i, z in enumerate(np.arange(start, end + step_length, step_length)):
        for j, x in enumerate(np.arange(start, end + step_length, step_length)):
            # Set the ray origin Q (x, y, z) with varying x and y
            Q = np.array([[x], [start], [z], [1]])
            
            # Coefficients of the quadratic equation at^2 + bt + c = 0
            a = magnitude(no_homo(d))**2
            b = 2 * np.dot(no_homo(d).T, no_homo(Q))[0][0]
            c = magnitude(no_homo(Q))**2 - radius**2

            # Discriminant
            discriminant = b**2 - 4*a*c

            # Check if the ray intersects the sphere
            if discriminant >= 0:
                canvas[i][j] = "x"
    
    # Print the 2D array
    for row in canvas:
        print(" ".join(row))


# v1 = vector(0,0,0,w)
# v2 = vector(0,1,0,w)
# v3 = vector(1,1,0,w)
# v4 = vector(1,0,0,w)
# u1 = vector(0,0,1,w)
# u2 = vector(0,1,1,w)
# u3 = vector(1,1,1,w)
# u4 = vector(1,0,1,w)
# 
# 
#

w=1
v1 = vector(0,0,0,w)
v2 = vector(0,1,0,w)
v3 = vector(1,1,0,w)
v4 = vector(1,0,0,w)
u1 = vector(0,0,1,w)
u2 = vector(0,1,1,w)
u3 = vector(1,1,1,w)
u4 = vector(1,0,1,w)
Scale = scale_matrix(2,2,2,w)
Rot = rotation_matrix_X(-0.25*np.pi)
square(cartesian(v1),cartesian(v2),cartesian(v3),cartesian(v4),cartesian(u1),cartesian(u2),cartesian(u3),cartesian(u4),'cyan')
transform_square(Scale)

#----------------------------------------------------------------------------------------
#Translations
# v1 = vector(1,1,1,w)
# v2 = vector(1,1,2,w)
# graph(v1, "r")
# graph(v2, "b")
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
#Rotations

# 
# square(cartesian(v1),cartesian(v2),cartesian(v3),cartesian(v4),cartesian(u1),cartesian(u2),cartesian(u3),cartesian(u4),'cyan')
# rot = rotation_matrix_X(-0.25*np.pi)
# rotate_square(-0.25*np.pi)
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
#Sphere

r = 2
w=1

# reccomeded step L:{ 0.19 < h < 1 }
# h = float(input("Enter separation distance  :"))
# domain = [-2,2]
# t=scale_matrix(5,5,5,w)
# # P(t) = Q + td0
# d = vector(0,1,0,w)
# Q = vector(-2,-2,-2,w)
# spawn_sphere(r)
# spawn_vector_field(h, d, t, -2, -2, -2, domain)
# draw_sphere_on_hit(Q, d, r, h, domain)
# side = "    x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x    "
# length = len(side)
# print(f"resolution = {length}x{length}")

#----------------------------------------------------------------------------------------

ax.view_init(30, 30)

plt.show()