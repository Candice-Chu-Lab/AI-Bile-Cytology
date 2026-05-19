### Install the Environemnt
virtualenv openslides
openslides\Scripts\activate

X-Anylabeling will be utilized for this task


### File description
- sliding_extraction.py: extarct patches from the whole slide image
- improved_filter_crtieria.py: helps filter the background patches (the patches that doesn't have much information)
- json_check.py: check if the label from X-Anylabeling is correct
- stratification.py: Given an annotation reult file that has the file format as in annotation_result.csv, construct k-fold cross validation dataset



### TODO
- For the folder that has tiles, we separate it into subfolder, with 400 images each.
- Is there a more efficient way to get the tiles besides the manual bounding box drawing phase?
https://github.com/deroneriksson/python-wsi-preprocessing/blob/master/docs/wsi-preprocessing-in-python/index.md


