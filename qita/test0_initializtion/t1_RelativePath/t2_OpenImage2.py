import matplotlib.pyplot as plt

img_path = r"img.png"
img = plt.imread(img_path)

fig = plt.figure('show picture')
# ax = fig.add_subplot(111)

plt.imshow(img)
plt.show()
