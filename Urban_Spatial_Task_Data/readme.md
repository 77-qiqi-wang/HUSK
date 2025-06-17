### Guidance on data usage and processing

All raw data is stored in the **'./Meta_data'** folder. You can process these files using the `preprocess_meta_data_nyc.py` or `preprocess_meta_data_chi.py` scripts to perform necessary cleaning and alignment. Once processed, the cleaned data will be saved into the **'./Processed_data'** folder.

Next, run the construct_USTP_Pointflow_XXX.py script to generate a dataset for spatio-temporal flow prediction, or use the construct_USTP_Event_XXX.py script to build a dataset for urban event prediction.

We storage them in the  **'./Urban_Spatial_Task'** directory.

The file information in each directory is as follows:

```
./Meta_data    Raw data set: taxi, bike, crime and 311 service event data.
./Processed_data   Aligned datasets: taxi, bike, human, crime and 311 service spatiotemporal dataset which are aligned with area, road and POI.
./Urban_Spatial_Task    The reformatted USTP dataset is now ready for use with downstream USTP models. 
```
