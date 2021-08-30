import numpy as np
import pandas as pd
from scipy.ndimage.measurements import center_of_mass #* Check if required
from scipy import interpolate #* Check if required
from scipy.ndimage.interpolation import shift#* Check if required
import warnings
import matplotlib.pyplot as plt
import os
import pickle
from FRDSolverClass import *

################################################################################
##############################################################################################################


class FRD_finder(object):
    """
    Class to find input unknown FRD images, compare them to a input/generated image set of known FRD,
    and plot them in a meaningful way."""
    
    def __init__(self, FRDlist, knownimagelist, imagetosolve, vartosolve,
                 ):
        """
        FRDlist is a list of each FRD used in the images to compare with an
        image with unknown FRD.
        
        knownimagelist is the list of images with the known FRDs corresponding
        to FRD list. Can pass None to generate this automatically
        
        imagetosolve is the image (matrix) with unknown FRD. Can pass None to
        read in the files.
        
        vartosolve is the variance corresponding to imagetosolve, of the
        same size. Can pass None to read in the files.
        """
        
        self.FRDlist = np.array(FRDlist)
                                  
        self.imagetosolve = np.array(imagetosolve)
        
        if vartosolve is not None:
            self.vartosolve = np.array(vartosolve)
        else:
            self.vartosolve = np.array(imagetosolve)
            

        self.knownimagelist = np.array(knownimagelist)
         
        #Change directory to match folder
        
        if len(np.shape(imagetosolve)) == 2: #imagetosolve having 2 dimensions for a 2d image 
            self.validpositions = 1
        else: #imagetosolve having 3 dimensions: 2d images for each position considered separately
            self.validpositions = len(imagetosolve) #E.G. number of 2D images
        
        if len(np.shape(self.knownimagelist)) == 3 and len(np.shape(self.imagetosolve)) == 3:
            if len(imagetosolve) == 1:
                self.knownimagelist = np.array([self.knownimagelist]) #Proper dimensions in the edge case
        
        self._test_types()
  
        self.residuallisttotal = None
        self.minFRDtotal = None
        self.FRD2dresiduals = None
        

    def _test_types(self):
        """Verifies that the inputs are of the proper form, throwing an exception if they are not.
        Includes more detailed error messages to help debugging."""
        if not isinstance(self.FRDlist,list) and not isinstance(self.FRDlist,np.ndarray):
            raise Exception('FRDlist should be a list or numpy array.')
        
        if not isinstance(self.knownimagelist,list) and not isinstance(self.knownimagelist,np.ndarray):
            raise Exception('knownimagelist should be a list or numpy array.')
            
            
            
        if not isinstance(self.imagetosolve,list) and not isinstance(self.imagetosolve,np.ndarray):
            raise Exception('imagetosolve should be a list or numpy array.') 
        if not isinstance(self.vartosolve,list) and not isinstance(self.vartosolve,np.ndarray):
            raise Exception('vartosolve should be a list or numpy array.') 
            

    def solve_FRD(self):
        residuallisttotal = {}
        minFRDtotal = {}
        FRD2dresiduals = {}
        
        print(len(np.shape(self.FRDlist)))
        print(len(np.shape(self.knownimagelist)))
        print(len(np.shape(self.imagetosolve)))
        
        
        testFRDFratio = FRDsolver(self.FRDlist, self.knownimagelist,
                                  self.imagetosolve,self.vartosolve)

        
        (residuallistFratio,minFRDFratio) = testFRDFratio.find_FRD_compare_positions(self.FRDlist, 
            self.knownimagelist, self.imagetosolve,self.vartosolve)

        residuallisttotal[str(positionindex)] = residuallistFratio
        minFRDtotal[str(positionindex)] = minFRDFratio
        FRD2dresiduals[str(positionindex)] = \
                       self.knownimagelist[self.FRDlist == minFRDFratio] - self.imagetosolve[positionindex]
        FRD2dresiduals[str(positionindex)] = \
                                FRD2dresiduals[str(positionindex)][0] #Remove type info
        
        self.residuallisttotal = residuallisttotal
        self.minFRDtotal = minFRDtotal
        self.FRD2dresiduals = FRD2dresiduals
        
        return (residuallisttotal,minFRDtotal,FRD2dresiduals)

    def plot_results_one(self,positionindex):
        
        positionindex = str(positionindex) #Convert input to the type keying the residuallist
        plt.plot(self.FRDlist,self.residuallisttotal[positionindex])
        plt.title(positionindex)
        plt.xlabel('FRD value')
        plt.ylabel('Sum of squares of residuals in mask')
        plt.show()

        maxvalue = (np.max(self.FRD2dresiduals[positionindex]))
        absminvalue = np.max(np.multiply(-1,(self.FRD2dresiduals[positionindex])))
        absmax = max(maxvalue,absminvalue)
        plt.imshow(self.FRD2dresiduals[positionindex],vmin=-absmax,vmax=absmax,origin='lower')
        plt.colorbar()
        plt.title('Position: {}'.format(positionindex))
        plt.show()    
    
    def plot_results_all(self):
        
        for positionindex in range(self.validpositions):
            plt.plot(self.FRDlist,self.residuallisttotal[str(positionindex)])
            plt.title(positionindex)
            plt.xlabel('FRD value')
            plt.ylabel('Sum of squares of residuals in mask')
            plt.show()
        
        #print(len(self.positionlist))
        for positionindex in range(self.validpositions):
            maxvalue = (np.max(self.FRD2dresiduals[str(positionindex)]))
            absminvalue = np.max(np.multiply(-1,(self.FRD2dresiduals[str(positionindex)])))
            absmax = max(maxvalue,absminvalue)
            plt.imshow(self.FRD2dresiduals[str(positionindex)],vmin=-absmax,vmax=absmax,origin='lower')
            plt.colorbar()
            plt.title('Position: {}'.format(positionindex))
            plt.show()