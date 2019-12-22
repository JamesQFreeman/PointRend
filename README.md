# PointRend
 An numpy-based implement of PointRend

This is an implement a PointRend function for Segmentation result refinement.
The paper can be find at https://arxiv.org/pdf/1912.08193.pdf

## Usage
copy the pointGenerate.py to your directory and you are ready to rock.
```
from pointGenerate import getpoint
my_mask = np.asarray(Image.open("tree_mask.jpg").resize((32,32)))
# convert this 3-channel binary mask to a 1-channel binary one
my_mask = my_mask[:,:,0]
# get the point, nearest_neighbor chose the sample points locations
points = getpoint(my_mask, k=2, beta = 0.95, nearest_neighbor=1)

# plot the result
points = list(zip(*points))
plt.imshow(my_mask,cmap="Purples")
plt.scatter(points[1],points[0],c='black',s=4)
```
## Some result
the original image and mask:

![mask](./tree_mask.jpg)
![img](./tree.jpg)

when the mask is 32*32

![mask size 32](./resolution=32.jpg)

when the mask is 64*64
![mask size 32](./resolution=64.jpg)

when the mask is 128*128
![mask size 32](./resolution=128.jpg)