#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:46:35 2021

@author: Mathew
"""

import numpy as np
import pandas as pd

import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from skimage import filters,measure
import scipy
import scipy.ndimage
import scipy.stats
from PIL import Image
import napari
from PIL import Image
from skimage.io import imread

# File information here
Pixel_size=117
image_width=int(856)
image_height=int(684)
scale=8


# Thresholds that need changing are here
eps_threshold=0.5
minimum_locs_threshold=20

overall_root=r"/Users/Mathew/Documents/Edinburgh Code/ONI_export/"       # Where to save all to
pathlist=[]

pathlist.append(r"/Users/Mathew/Documents/Edinburgh Code/ONI_export/")
# pathlist.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/PSD95 Figure/Datasets/2021-06-08_New_Dataset_5uM/CA1_SO/")
# pathlist.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/PSD95 Figure/Datasets/2021-06-08_New_Dataset_5uM/DG_CL_LB/")
# pathlist.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/PSD95 Figure/Datasets/2021-06-08_New_Dataset_5uM/DG_PL_LB/")
# pathlist.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/PSD95 Figure/Datasets/2021-06-08_New_Dataset_5uM/CA1_SRP_1/")
# pathlist.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/PSD95 Figure/Datasets/2021-06-08_New_Dataset_5uM/CA3_SO_RB/")
# pathlist.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/PSD95 Figure/Datasets/2021-06-08_New_Dataset_5uM/CA3_SLU/")
# pathlist.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/PSD95 Figure/Datasets/2021-06-08_New_Dataset_5uM/DG_CL_RB/")
# pathlist.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/PSD95 Figure/Datasets/2021-06-08_New_Dataset_5uM/CA3_SO/")
filename_contains='only.csv'   # This is the name of the SR file containing the localisations




# DBSCAN 
def cluster(coords):
     db = DBSCAN(eps=eps_threshold, min_samples=minimum_locs_threshold).fit(coords)
     labels = db.labels_
     n_clusters_ = len(set(labels)) - (1 if-1 in labels else 0)  # This is to calculate the number of clusters.
     print('Estimated number of clusters: %d' % n_clusters_)
     return labels

# Generate figures
def gkern(l,sigx,sigy):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    # ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx)/np.square(sigx) + np.square(yy)/np.square(sigy)) )

    return kernel/np.sum(kernel)

# Generate the super resolution image with points equating to the cluster number
def generate_SR(coords,clusters):
    SR_plot=np.zeros((image_width*scale,image_height*scale),dtype=float)
    SR_plot_clu=np.zeros((image_width*scale,image_height*scale),dtype=float)
    j=0
    for i in clusters:
        if i>-1:
            
            xcoord=coords[j,0]
            ycoord=coords[j,1]
            scale_xcoord=round(xcoord*scale)
            scale_ycoord=round(ycoord*scale)
            # if(scale_xcoord<image_width and scale_ycoord<image_height):
            SR_plot[scale_xcoord,scale_ycoord]+=1
            SR_plot_clu[scale_xcoord,scale_ycoord]=i
            
        j+=1
    return SR_plot,SR_plot_clu

# Generate SR image with width = precision
def generate_SR_prec(coords,clusters):

    SR_prec_plot=np.zeros((image_width*scale,image_height*scale),dtype=float)
    SR_prec_clu_plot=np.zeros((image_width*scale,image_height*scale),dtype=float)
    SR_prec_clu_counter=np.zeros((image_width*scale,image_height*scale),dtype=float)
    j=0
    for i in clusters:
        if i>-1:
            precisionx=precsx[j]
            precisiony=precsy[j]
            
            xcoord=coords[j,0]
            ycoord=coords[j,1]
            scale_xcoord=round(xcoord*scale)
            scale_ycoord=round(ycoord*scale)
            sigmax=8*precisionx
            sigmay=8*precisiony
            gauss_to_add=gkern(20,sigmax,sigmay)
            # if((scale_xcoord+10)<image_width and (scale_ycoord+10)<image_height and (scale_xcoord-10)>0 and (scale_ycoord-10)>0):
            SR_prec_plot[scale_xcoord-10:scale_xcoord+10,scale_ycoord-10:scale_ycoord+10]=SR_prec_plot[scale_xcoord-10:scale_xcoord+10,scale_ycoord-10:scale_ycoord+10]+gauss_to_add
           
           
           
            half_maximum=gauss_to_add.max()/2.0
            gauss_FWHM=gauss_to_add>=half_maximum
            gauss_to_add_clu=gauss_FWHM*i
            SR_prec_clu_plot[scale_xcoord-10:scale_xcoord+10,scale_ycoord-10:scale_ycoord+10]=SR_prec_clu_plot[scale_xcoord-10:scale_xcoord+10,scale_ycoord-10:scale_ycoord+10]+gauss_to_add_clu
            SR_prec_clu_counter[scale_xcoord-10:scale_xcoord+10,scale_ycoord-10:scale_ycoord+10]=SR_prec_clu_counter[scale_xcoord-10:scale_xcoord+10,scale_ycoord-10:scale_ycoord+10]+gauss_FWHM
        
        
        
            
        j+=1
    SR_prec_clu_plot=np.divide(SR_prec_clu_plot, SR_prec_clu_counter,where=SR_prec_clu_counter!=0)
    return SR_prec_plot,SR_prec_clu_plot

# Try labelling with the usual plugin- will help get the sizes etc. 

def threshold_image_fixed(input_image,threshold_number):
    threshold_value=threshold_number   
    binary_image=input_image>threshold_value

    return binary_image

# Label and count the features in the thresholded image:
def label_image(input_image):
    labelled_image=measure.label(input_image)
    number_of_features=labelled_image.max()
 
    return number_of_features,labelled_image

def analyse_labelled_image(labelled_image,original_image):
    measure_image=measure.regionprops_table(labelled_image,intensity_image=original_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length','mean_intensity','max_intensity'))
    measure_dataframe=pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

def view_image(*input_image):
    with napari.gui_qt():
        
            viewer = napari.Viewer()
            for inputs in input_image:
                viewer.add_image(inputs)
 


def load_image(toload):
    
    image=imread(toload)
    
    return image

# z-project the image

def z_project(image):
    
    mean_int=np.mean(image,axis=0)
  
    return mean_int

# To store overall medians/means etc. 
Output_all_cases = pd.DataFrame(columns=['Path','Number_of_clusters','Points_per_cluster_mean','Points_per_cluster_SD','Points_per_cluster_med',
                                       'Area_mean','Area_sd','Area_med','Length_mean','Length_sd','Length_med','Ratio_mean','Ratio_sd','Ratio_med'])
for path in pathlist:
    #  Load the data
    
    for root, dirs, files in os.walk(path):
        for name in files:
                if filename_contains in name:
                    resultsname = name
                    print(name)
    
    
    
    data_df = pd.read_csv(path+resultsname)
    
    # Get the correct rows out
    coords = np.array(list(zip(data_df['X (pix)'],data_df['Y (pix)'])))
    precsx= np.array(data_df['X precision (pix)'])
    precsy= np.array(data_df['Y precision (pix)'])
    xcoords=np.array(data_df['X (pix)'])
    ycoords=np.array(data_df['Y (pix)'])
    
    # Peform cluster analysis on the data
    clusters=cluster(coords)
    
    SR,SR_Clu=generate_SR(coords,clusters)
    SR_prec,SR_prec_clu_plot=generate_SR_prec(coords,clusters)
    
    
    
    
    
    imsr = Image.fromarray(SR_prec)
    imsr.save(path+'SR_from_python.tif')
    
    imsr = Image.fromarray(SR_prec_clu_plot)
    imsr.save(path+'SR_clusters_FWHM.tif')
    
    imsr = Image.fromarray(SR_Clu)
    imsr.save(path+'SR_clusters.tif')
    
    ims = Image.fromarray(SR)
    ims.save(path+'SR_points.tif')
    
    
    
    
    
    # How many localisations per cluster?
    clu=clusters.tolist()
    maximum=clusters.max()      # Total number of clusters. 
    cluster_contents=[]         # Make a list to store the number of clusters in
    
    for i in range(0,maximum):
        n=clu.count(i)     # Count the number of times that the cluster number i is observed
        cluster_contents.append(n)  # Add to the list. 
    
        

    plt.hist(cluster_contents, bins = 20,range=[0,200], rwidth=0.9,color='#ff0000')
    plt.xlabel('Localisations per cluster',size=20)
    plt.ylabel('Number of Features',size=20)
    plt.title('Cluster localisations',size=20)
    plt.savefig(path+"Localisations.pdf")
    plt.show()

    cluster_arr=np.array(cluster_contents)
    
    median_locs=np.median(cluster_arr)
    mean_locs=cluster_arr.mean()
    std_locs=cluster_arr.std()
    
    
    # This is to perform some distance analysis etc. 
    
    # binary=threshold_image_fixed(SR_prec,0.01)
    labelled=SR_prec_clu_plot.astype('int')
    
    measurements=analyse_labelled_image(labelled,SR_prec)
    
    
    # Make and save histograms
    
    areas=measurements['area']*((Pixel_size/(scale*1000))**2)
    plt.hist(areas, bins = 20,range=[0,0.1], rwidth=0.9,color='#ff0000')
    plt.xlabel('Area (\u03bcm$^2$)',size=20)
    plt.ylabel('Number of Features',size=20)
    plt.title('Cluster area',size=20)
    plt.savefig(path+"Area.pdf")
    plt.show()
    
    median_area=areas.median()
    mean_area=areas.mean()
    std_area=areas.std()
    
    
    
    length=measurements['major_axis_length']*((Pixel_size/8))
    plt.hist(length, bins = 20,range=[0,500], rwidth=0.9,color='#ff0000')
    plt.xlabel('Length (nm)',size=20)
    plt.ylabel('Number of Features',size=20)
    plt.title('Cluster lengths',size=20)
    plt.savefig(path+"Lengths.pdf")
    plt.show()

    median_length=length.median()
    mean_length=length.mean()
    std_length=length.std()
    
    ratio=measurements['minor_axis_length']/measurements['major_axis_length']
    plt.hist(ratio, bins = 50,range=[0,1], rwidth=0.9,color='#ff0000')
    plt.xlabel('Eccentricity',size=20)
    plt.ylabel('Number of Features',size=20)
    plt.title('Cluster Eccentricity',size=20)
    plt.savefig(path+"Ecc.pdf")
    plt.show()
    
    median_ratio=ratio.median()
    mean_ratio=ratio.mean()
    std_ratio=ratio.std()
    
    
    measurements['Eccentricity']=ratio
    measurements['Number_of_locs']=cluster_contents
    measurements.to_csv(path + '/' + 'Metrics.csv', sep = '\t')
    
    Output_overall = pd.DataFrame(columns=['xw','yw','cluster'])
    
    Output_overall['xw']=xcoords
    
    Output_overall['yw']=ycoords
    
    Output_overall['cluster']=clusters
    
    Output_overall.to_csv(path + '/' + 'all.csv', sep = '\t')    

    Output_all_cases = Output_all_cases.append({'Path':path,'Number_of_clusters':maximum,'Points_per_cluster_mean':mean_locs,'Points_per_cluster_SD':std_locs,'Points_per_cluster_med':median_locs,
                                        'Area_mean':mean_area,'Area_sd':std_area,'Area_med':median_area,'Length_mean':mean_length,'Length_sd':std_length,'Length_med':median_length,
                                        'Ratio_mean':mean_ratio,'Ratio_sd':std_ratio,'Ratio_med':median_ratio},ignore_index=True)

Output_all_cases.to_csv(overall_root + '/' + 'all_metrics.csv', sep = '\t')









