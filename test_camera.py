import airsim
import numpy as np
import cv2

IMAGE_WIDTH  = 160
IMAGE_HEIGHT = 120

print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected!")

responses = client.simGetImages([
    airsim.ImageRequest('front_rgb', airsim.ImageType.Scene, False, False),
    airsim.ImageRequest('front_depth', airsim.ImageType.DepthPlanar, True, False),
])

# RGB decode + resize
rgb_response = responses[0]
rgb_1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
rgb = rgb_1d.reshape(rgb_response.height, rgb_response.width, 3)
rgb = cv2.resize(rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))

# Depth decode + resize
depth_response = responses[1]
depth = np.array(depth_response.image_data_float, dtype=np.float32)
depth = depth.reshape(depth_response.height, depth_response.width)
depth = cv2.resize(depth, (IMAGE_WIDTH, IMAGE_HEIGHT))

print(f"AirSim RGB   raw size: {rgb_response.width} x {rgb_response.height}")
print(f"AirSim Depth raw size: {depth_response.width} x {depth_response.height}")
print(f"RGB   after resize: {rgb.shape}")
print(f"Depth after resize: {depth.shape}")

if rgb.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3) and depth.shape == (IMAGE_HEIGHT, IMAGE_WIDTH):
    print("\nALL OK - both images correctly sized for navigation pipeline!")
else:
    print("\nFAILED - shapes still wrong")