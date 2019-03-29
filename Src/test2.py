
import numpy as np
from PIL import Image

Iamb = Image.open('./Example_Img/Buddha/photometric_sample_raw_0001.png')
# data = array(Iamb)
print(type(Iamb))
in_data = np.asarray(Iamb, dtype=np.uint16)
print(in_data)

# import png
# import numpy as np

# reader = png.Reader('./Example_Img/Buddha/photometric_sample_raw_0001.png')
# pngdata = reader.read()
# px_array = np.stack( map( np.uint16, pngdata[2] ) )
# print( px_array )

# import png
# import numpy as np
# reader = png.Reader('./Example_Img/Buddha/photometric_sample_raw_0001.png')
# data = reader.asDirect()
# pixels = data[2]
# image = []
# for row in pixels:
#   row = np.asarray(row)
#   row = np.reshape(row, [-1, 3])
#   image.append(row)
# image = np.stack(image, 1)
# print(image.dtype)
# print(image.shape)

# import numpy as np

 

# import scipy.misc

# from PIL import Image

# tif16 = scipy.misc.imread('./Example_Img/Buddha/photometric_sample_raw_0001.png') 
# print(np.shape(tif16),type(tif16[0,0]))

# print(tif16)