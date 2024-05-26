from ultralytics import YOLO
import cv2

model_path = r'D:\Hackathon\pythonProject\runs\segment\train2\weights\last.pt'
image_path = r'D:\Hackathon\pythonProject\runs\segment\train2\imh1.jpg'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)
results = model(img)
print(results)

# Initialize an empty list to store masks if any
all_masks = []

for result in results:
    if hasattr(result, 'masks') and result.masks is not None:
        for j, mask in enumerate(result.masks.data):
            mask = mask.numpy() * 255
            mask = cv2.resize(mask, (W, H))
            all_masks.append(mask)
    else:
        print("No masks were detected.")

# Save all masks to separate files if any masks were detected
if all_masks:
    for i, mask in enumerate(all_masks):
        cv2.imwrite(f'./output_{i}.png', mask)
else:
    print("No masks to save.")
