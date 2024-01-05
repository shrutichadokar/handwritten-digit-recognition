
from tensorflow.keras.datasets import mnist

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


sample = 1
image = X_train[sample]

fig = plt.figure
plt.imshow(image,cmap='gray')
plt.show()


# In[9]:


fig = plt.figure
plt.imshow(image,cmap='gray_r')
plt.show()

num = 10
images = X_train[:num]
labels = Y_train[:num]
print(images)
print(labels)

num = 10
images = X_train[:num]
labels = Y_train[:num]

num_row = 2
num_col = 5

fig,axes = plt.subplots(num_row,num_col,figsize = (1.5*num_col,2*num_row))

for i in range(num):
    ax = axes[i//num_col,i%num_col]
    ax.imshow(images[i],cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))

plt.tight_layout()
plt.show()

