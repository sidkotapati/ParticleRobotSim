import pymunk
import numpy as np
import pygame
import pymunk.pygame_util

# Constants
rod_length = 0.1
rod_width = 0.04
radius = 0.05
boundary_radius = 1.0

# Display setup
display_window = (800, 600)
display_scale = 200
pygame.init()
screen = pygame.display.set_mode(display_window)
clock = pygame.time.Clock()

# Pymunk display options
pymunk.pygame_util.positive_y_is_up = True
display_offset = np.array(display_window) / (2 * display_scale)
draw_options = pymunk.pygame_util.DrawOptions(screen)
draw_options.transform = pymunk.Transform.scaling(display_scale).translated(*display_offset)
draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

# Create space
space = pymunk.Space()
space.gravity = (0, 0)

# Lists to store robot parts and boundary segments
roller_seg = []
roller_left = []
roller_right = []

# Create boundary
boundary_body = pymunk.Body(body_type=pymunk.Body.STATIC)
space.add(boundary_body)

boundary_points = []
num_segments = 32
for i in range(num_segments):
    angle = 2 * np.pi * i / num_segments
    x = boundary_radius * np.cos(angle)
    y = boundary_radius * np.sin(angle)
    boundary_points.append((x, y))

# Create boundary segments
for i in range(num_segments):
    p1 = boundary_points[i]
    p2 = boundary_points[(i + 1) % num_segments]
    segment = pymunk.Segment(boundary_body, p1, p2, 0.01)
    segment.friction = 1000
    segment.elasticity = 0.0
    space.add(segment)

def apply_torque(body, magnitude, r=1):
    body.apply_force_at_local_point((0, magnitude / (2*r)), (r, 0))
    body.apply_force_at_local_point((0, -magnitude / (2*r)), (-r, 0))

# Create robots
num_bots = 40
for i in range(num_bots):
    while True:
        pos = np.random.uniform(-0.7, 0.7, 2)
        if not any(np.linalg.norm(pos - b.position) <= rod_length + radius for b in roller_left + roller_right):
            break
            
    angle = np.random.uniform(0, 2*np.pi)
    rhat = np.array([np.cos(angle), np.sin(angle)])

    _filter = pymunk.ShapeFilter(i)

    # Middle segment
    segbody = pymunk.Body()
    segshape = pymunk.Poly(segbody, [[a*rod_length, b*rod_width] for a in (-1, 1) for b in (-1, 1)])
    segshape.mass = 1
    segshape.friction = 1000
    segshape.color = (192, 192, 192, 255)
    segshape.filter = _filter

    segbody.position = tuple(pos)
    segbody.angle = angle

    space.add(segbody, segshape)
    roller_seg.append(segbody)

    # Create wheels
    for side, arr in zip([-1, 1], [roller_left, roller_right]):
        cpos = pos + side * rod_length * rhat
        circbody = pymunk.Body()
        circshape = pymunk.Circle(circbody, radius, (0, 0))
        circshape.mass = 1
        circshape.friction = 1000
        circshape.color = (128, 128, 128, 255)
        circshape.filter = _filter

        circbody.position = tuple(cpos)
        circbody.angle = angle

        arr.append(circbody)

        joint = pymunk.PivotJoint(segbody, circbody, tuple(cpos))
        space.add(circbody, circshape, joint)

# Actuation parameters
actuation_left = [0.01 for _ in range(num_bots)]
actuation_right = [-0.01 for _ in range(num_bots)]

# Simulation loop
time = 0
time_steps = 10000
dt = 0.02

for step in range(time_steps):
    for _ in range(5):
        # Update system
        for act, body in zip(actuation_left + actuation_right, roller_left + roller_right):
            apply_torque(body, act)
            body.angular_velocity *= 0.97
            body.apply_force_at_world_point(-0.7 * body.velocity, body.position)

        # Uniform center force
        for body in roller_seg:
            body.apply_force_at_world_point(-0.01 * body.position, body.position)
        
        space.step(dt)
        time += dt

    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    # Draw
    screen.fill((255, 255, 255))
    space.debug_draw(draw_options)
    pygame.display.update()
    clock.tick(60)

    print(f"{int(clock.get_fps())} fps / time = {time:.2f}", end="\r")