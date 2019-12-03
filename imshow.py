from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
i = 35
path = 'array_save.npz'
data = np.load(path)
# warped_moving = Image.fromarray(data['warped_moving'][:,:,i])
# warped_moving0 = Image.fromarray(data['warped_moving0'][:,:,i])
# warped_moving1 = Image.fromarray(data['warped_moving1'][:,:,i])
# warped_moving2 = Image.fromarray(data['warped_moving2'][:,:,i])
# warped_moving3 = Image.fromarray(data['warped_moving3'][:,:,i])
# warped_moving4 = Image.fromarray(data['warped_moving4'][:,:,i])
# warped_moving5 = Image.fromarray(data['warped_moving5'][:,:,i])
# warped_moving6 = Image.fromarray(data['warped_moving6'][:,:,i])
# warped_moving7 = Image.fromarray(data['warped_moving7'][:,:,i])
# warped_moving8 = Image.fromarray(data['warped_moving8'][:,:,i])
# warped_moving9 = Image.fromarray(data['warped_moving9'][:,:,i])
# warped_moving10 = Image.fromarray(data['warped_moving10'][:,:,i])
# image_fixed = Image.fromarray(data['image_fixed'][:,:,i])


warped_moving = data['arr_0'][:,:,i]
warped_moving0 = data['arr_1'][:,:,i]
warped_moving1 = data['arr_2'][:,:,i]
warped_moving2 = data['arr_3'][:,:,i]
warped_moving3 = data['arr_4'][:,:,i]
warped_moving4 = data['arr_5'][:,:,i]
warped_moving5 = data['arr_6'][:,:,i]
warped_moving6 = data['arr_7'][:,:,i]
warped_moving7 = data['arr_8'][:,:,i]
warped_moving8 = data['arr_9'][:,:,i]
warped_moving9 = data['arr_10'][:,:,i]
warped_moving10 = data['arr_11'][:,:,i]
image_fixed = data['arr_12'][:,:,i]

# matplotlib.image.imsave('name.png', data['arr_1'][:,:,i])

plt.figure(1)
plt.subplot(4,4,1)
plt.imshow(image_fixed,cmap='gray')
plt.subplot(4,4,5)
plt.imshow(warped_moving0,cmap='gray')
plt.subplot(4,4,6)
plt.imshow(warped_moving1,cmap='gray')
plt.subplot(4,4,7)
plt.imshow(warped_moving2,cmap='gray')
plt.subplot(4,4,8)
plt.imshow(warped_moving3,cmap='gray')
plt.subplot(4,4,9)
plt.imshow(warped_moving4,cmap='gray')
plt.subplot(4,4,10)
plt.imshow(warped_moving5,cmap='gray')
plt.subplot(4,4,11)
plt.imshow(warped_moving6,cmap='gray')
plt.subplot(4,4,12)
plt.imshow(warped_moving7,cmap='gray')
plt.subplot(4,4,13)
plt.imshow(warped_moving8,cmap='gray')
plt.subplot(4,4,14)
plt.imshow(warped_moving9,cmap='gray')
plt.subplot(4,4,15)
plt.imshow(warped_moving10,cmap='gray')
plt.subplot(4,4,16)
plt.imshow(warped_moving,cmap='gray')
plt.show()
plt.figure(2)
plt.subplot(1,3,1)
plt.imshow(image_fixed,cmap='gray')
plt.title("Fixed", fontsize=15, pad=20)
plt.subplot(1,3,2)
plt.imshow(warped_moving1,cmap='gray')
plt.title("Moving", fontsize=15, pad=20)
plt.subplot(1,3,3)
plt.imshow(warped_moving,cmap='gray')
plt.title("Warp_Moving", fontsize=15, pad=20)
plt.show()