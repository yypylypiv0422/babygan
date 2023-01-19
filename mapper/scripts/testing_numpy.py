import numpy as np

# latent_codes_origin = np.load(
#     '/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/hairai/HairCLIP-main/code/alice-muriithi-6qfxmy3657c-unsplash_01.npy')
# print(latent_codes_origin.shape)
import string
def sortString(str) :
    str = ''.join(sorted(str))

    print(str)
    return str
stringss="aaba AA bb cc d $ % &"
oo=stringss.lower()
new_str=sortString(oo)
print(stringss.lower())
print(stringss.lower())
all_alpha=list(string.ascii_lowercase)
dict={}
count=1
j=''
discount = 0
lis = []

for i in new_str:
    # print(i)
    # print(i.isalpha())
    # j = ''
    # count=0
    if i.isalpha():

       if i==j:
           count=count+1
           print(count)
           lis.append(i)
       else:
           lis=[]

           count=1
           discount=0

       # if len(lis)>=1:
       #     discount=count

       # lis.append(i)

       j=i


       # print(i,'llll')

# importing Counter module
from collections import Counter
# list(string.split(" "))
input_list = new_str

# creating a list with the keys
items = Counter(input_list).keys()
print("No of unique items in the list are:", items)