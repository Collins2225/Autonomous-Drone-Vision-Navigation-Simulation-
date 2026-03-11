# Utilities Module - Helper functions and visualizers

import airsim
import numpy as np
import cv2

client = airsim.MultirotorClient()
client.confirmConnection()

responses = client.simGetImages([
    airsim.ImageRequest('front_rgb', airsim.ImageType.Scene, False, False),
    airsim.ImageRequest('front_depth', airsim.ImageType.DepthPlanar, True, False),
])

rgb_raw = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
rgb = cv2.imdecode(rgb_raw, cv2.IMREAD_COLOR)
if rgb is not None:
    rgb = cv2.resize(rgb, (160, 120))

depth = np.array(responses[1].image_data_float, dtype=np.float32)

print('RGB response size:  ', len(responses[0].image_data_uint8), 'bytes')
print('RGB decoded shape:  ', rgb.shape if rgb is not None else 'FAILED')
print('Depth array length: ', len(depth))
print('Expected depth size:', 160 * 120)
print('ALL OK' if rgb is not None and len(depth) > 0 else 'STILL BROKEN')